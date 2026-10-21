"""Train-on-training-worksheet, test-on-validation-sample comparison of sentiment models."""
import random
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

from src.evaluation import LABELS, bootstrap_ci, macro_f1, summary

LABEL_IDS = {label: i for i, label in enumerate(LABELS)}

LLM_INSTRUCTIONS = (
    "You label farmers' feedback about Indian government welfare schemes. "
    "Reply with exactly one word: Positive, Negative or Neutral.\n"
    "Positive = mainly expresses satisfaction, benefit, relief or approval.\n"
    "Negative = mainly expresses a problem, delay, frustration or complaint.\n"
    "Neutral = factual with no clear tone, or genuinely mixed with no dominant tone."
)

ABSA_PHRASES = {
    "Timeliness": "timely payment",
    "Amount_of_Support": "amount of support",
    "Enrollment_Ease": "enrollment process",
    "Official_Behavior": "officials",
    "Clarity_of_Info": "information",
    "Overall_Impact": "overall impact",
}


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def free_gpu(*objects):
    """Release GPU memory between models; without this, consecutive models fill a small GPU."""
    import gc
    import torch
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stratified_fraction(texts, labels, fraction, seed):
    frame = pd.DataFrame({"text": texts.values, "label": labels.values})
    part = frame.groupby("label", group_keys=False).sample(frac=fraction, random_state=seed)
    return part["text"], part["label"]


def tfidf_logreg(train_x, train_y, test_x, seed=42):
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1),
                          LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
    return model.fit(train_x, train_y).predict(test_x)


def setfit_predict(train_x, train_y, test_x, seed, backbone, iterations=5):
    """iterations = contrastive pairs generated per training text (SetFit default is 20, meant for few dozen labels)."""
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    set_seed(seed)
    model = SetFitModel.from_pretrained(backbone)
    data = Dataset.from_dict({"text": list(train_x), "label": [LABEL_IDS[y] for y in train_y]})
    with tempfile.TemporaryDirectory() as checkpoints:
        args = TrainingArguments(output_dir=checkpoints, batch_size=16, num_epochs=1, num_iterations=iterations, seed=seed, save_strategy="no")
        Trainer(model=model, args=args, train_dataset=data).train()
    preds = np.array([LABELS[int(i)] for i in model.predict(list(test_x))])
    model.model_body.to("cpu")
    free_gpu(model)
    return preds


def finetune_predict(train_x, train_y, test_x, seed, backbone, epochs=4, lr=2e-5, batch_size=16, max_len=128):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(seed)
    dev = device()
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL_IDS, ignore_mismatched_sizes=True,
    ).to(dev)

    x, y = list(train_x), torch.tensor([LABEL_IDS[v] for v in train_y])
    counts = torch.bincount(y, minlength=len(LABELS)).float().clamp(min=1)
    loss_fn = torch.nn.CrossEntropyLoss(weight=(counts.sum() / (len(LABELS) * counts)).to(dev))
    if dev == "cuda":
        import bitsandbytes as bnb  # 8-bit optimiser states so base-size models train on a 4 GB GPU
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else torch.autocast("cpu", enabled=False)
    steps = epochs * ((len(x) + batch_size - 1) // batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    generator = torch.Generator().manual_seed(seed)

    model.train()
    for _ in range(epochs):
        for batch in torch.randperm(len(x), generator=generator).split(batch_size):
            enc = tokenizer([x[i] for i in batch], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(dev)
            with autocast:
                logits = model(**enc).logits
            loss = loss_fn(logits.float(), y[batch].to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    model.eval()
    preds, texts = [], list(test_x)
    with torch.no_grad():
        for start in range(0, len(texts), 64):
            enc = tokenizer(texts[start:start + 64], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(dev)
            with autocast:
                preds += model(**enc).logits.argmax(-1).tolist()
    model.to("cpu")
    free_gpu(model, optimizer)
    return np.array([LABELS[i] for i in preds])


def few_shot_examples(train_x, train_y, per_class, seed):
    frame = pd.DataFrame({"text": train_x.values, "label": train_y.values})
    picked = pd.concat([g.sample(n=min(per_class, len(g)), random_state=seed) for _, g in frame.groupby("label")])
    return picked.sample(frac=1, random_state=seed)


def llm_messages(text, examples):
    messages = [{"role": "system", "content": LLM_INSTRUCTIONS}]
    for row in examples.itertuples():
        messages += [{"role": "user", "content": row.text}, {"role": "assistant", "content": row.label}]
    return messages + [{"role": "user", "content": text}]


def parse_label(reply):
    reply = reply.strip().lower()
    for label in LABELS:
        if reply.startswith(label.lower()):
            return label
    hits = [label for label in LABELS if label.lower() in reply]
    return hits[0] if len(hits) == 1 else None


def llm_predict(test_x, model_name, examples, four_bit=False, dtype="auto"):
    """Greedy (deterministic) decoding; unparsable replies count as Neutral and are reported."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if four_bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant, device_map="auto")
    else:
        if dtype == "auto":
            dtype = "float16" if device() == "cuda" else "float32"
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype)).to(device())
    model.eval()
    preds, unparsed = [], 0
    for text in test_x:
        prompt = tokenizer.apply_chat_template(llm_messages(text, examples), add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device())
        with torch.no_grad():
            out = model.generate(**prompt, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        label = parse_label(tokenizer.decode(out[0, prompt["input_ids"].shape[1]:], skip_special_tokens=True))
        unparsed += label is None
        preds.append(label or "Neutral")
    del model, prompt, out  # drop the last references before clearing, or the next LLM cannot fit
    free_gpu()
    return np.array(preds), unparsed


def absa_predict(texts, phrases, model_name):
    from transformers import pipeline
    classifier = pipeline("text-classification", model=model_name, tokenizer=model_name, device=device())
    out = classifier([{"text": t, "text_pair": p} for t, p in zip(texts, phrases)], batch_size=16)
    return np.array([o["label"].capitalize() for o in out])


def majority_vote(runs):
    stacked = np.vstack(runs)
    return np.array([pd.Series(col).mode().iloc[0] for col in stacked.T])


def mcnemar(y_true, pred_a, pred_b):
    correct_a, correct_b = np.asarray(pred_a) == y_true, np.asarray(pred_b) == y_true
    b, c = int((correct_a & ~correct_b).sum()), int((~correct_a & correct_b).sum())
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def method_row(name, kind, y_true, runs):
    y_true = np.asarray(y_true)
    pred = runs[0] if len(runs) == 1 else majority_vote(runs)
    row = {"method": name, "type": kind, "seeds": len(runs), **summary(y_true, pred)}
    row["macro_f1_ci95"] = "{} - {}".format(*bootstrap_ci(y_true, pred, macro_f1))
    if len(runs) > 1:
        f1s = [macro_f1(y_true, r) for r in runs]
        accs = [accuracy_score(y_true, r) for r in runs]
        row["macro_f1_seed_mean_sd"] = f"{np.mean(f1s):.3f} ± {np.std(f1s, ddof=1):.3f}"
        row["accuracy_seed_mean_sd"] = f"{np.mean(accs):.3f} ± {np.std(accs, ddof=1):.3f}"
    return row, pred
