"""Compare sentiment models: zero-shot baselines, models trained on Training_Worksheet.xlsx, and LLMs.

Test set = the human labels in Annotation_Worksheet.xlsx (stratified random 20% sample, never used for training).
Training set = the human labels in Training_Worksheet.xlsx (script 11; disjoint from the test set).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import MASTER_FILE, MASTER_SHEET, OUTPUT_DIR
from src.evaluation import LABELS, attach_master, confusion, inter_annotator_kappa, per_class, read_worksheet, summary
from src.excel_style import save_workbook
from src import model_comparison as mc

ZERO_SHOT = {"VADER + agri lexicon": "vader_on_feedback_label", "RoBERTa Twitter (zero-shot)": "transformer_label"}
ABSA_MASTER = {
    "Timeliness": "absa_timeliness_label", "Amount_of_Support": "absa_amount_label", "Enrollment_Ease": "absa_enrollment_ease_label",
    "Official_Behavior": "absa_official_behavior_label", "Clarity_of_Info": "absa_clarity_label", "Overall_Impact": "absa_impact_label",
}


def stop(message):
    sys.exit(f"Stopped: {message}")


def load(path, master, extra_cols):
    sheet, invalid = read_worksheet(path, ["Human_Label"])
    if invalid:
        stop(f"{path.name} has labels other than Positive/Neutral/Negative: {invalid}")
    sheet, mismatched = attach_master(sheet, master, extra_cols)
    if mismatched:
        stop(f"{path.name}: {mismatched} rows no longer match the master results file.")
    return sheet[sheet["Human_Label"].notna()].reset_index(drop=True)


def timed(label, fn):
    start = time.time()
    result = fn()
    print(f"  {label}: {time.time() - start:.0f}s")
    return result


def learning_curve(args, train, test, y_test):
    rows = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        for seed in range(min(3, args.seeds)):
            x, y = mc.stratified_fraction(train["Feedback_Text"], train["Human_Label"], fraction, seed)
            runs = {"TF-IDF logistic regression": lambda: mc.tfidf_logreg(x, y, test["Feedback_Text"], seed)}
            if args.curve_setfit and not args.skip_setfit:
                runs[f"SetFit ({args.setfit_backbone.split('/')[-1]})"] = lambda: mc.setfit_predict(x, y, test["Feedback_Text"], seed, args.setfit_backbone)
            for name, run in runs.items():
                rows.append({"method": name, "train_fraction": fraction, "train_n": len(x), "seed": seed,
                             "macro_f1": round(mc.macro_f1(y_test, run()), 3)})
    curve = pd.DataFrame(rows)
    return curve.groupby(["method", "train_fraction", "train_n"], as_index=False)["macro_f1"].agg(["mean", "std"]).round(3)


def absa_comparison(path, master, model_name):
    sheet, invalid = read_worksheet(path, list(ABSA_MASTER))
    if invalid:
        stop(f"{path.name} has invalid labels: {invalid}")
    sheet, mismatched = attach_master(sheet, master, list(ABSA_MASTER.values()))
    if mismatched:
        stop(f"{path.name}: {mismatched} rows no longer match the master results file.")
    cells = [(row.Feedback_Text, aspect, getattr(row, aspect), getattr(row, column))
             for row in sheet.itertuples() for aspect, column in ABSA_MASTER.items() if pd.notna(getattr(row, aspect))]
    if not cells:
        return None
    frame = pd.DataFrame(cells, columns=["text", "aspect", "human", "pipeline"])
    frame["pipeline"] = frame["pipeline"].fillna("Not detected")
    frame["deberta_absa"] = mc.absa_predict(frame["text"], frame["aspect"].map(mc.ABSA_PHRASES), model_name)
    rows = []
    for method, col in [("Keyword ABSA + VADER (current pipeline)", "pipeline"), (f"DeBERTa ABSA ({model_name.split('/')[-1]})", "deberta_absa")]:
        rows.append({"method": method, "aspect_mentions": len(frame), "detected": int((frame[col] != "Not detected").sum()),
                     **summary(frame["human"], frame[col])})
    return pd.DataFrame(rows)


def load_cache(path, key):
    """Predictions from earlier runs, reused when the test rows and training labels are unchanged."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf8"))
    if data.get("key") != key:
        print("Test rows or training labels changed since the cached run - recomputing every model")
        return {}
    return data["methods"]


def save_cache(path, key, cache):
    path.write_text(json.dumps({"key": key, "methods": cache}), encoding="utf8")


def external_predictions(spec, test):
    name, _, csv_path = spec.partition("=")
    preds = pd.read_csv(csv_path).set_index("Source_Excel_Row")["label"].astype(str)
    missing = set(test["Source_Excel_Row"]) - set(preds.index)
    if missing:
        stop(f"{csv_path} lacks predictions for {len(missing)} test rows")
    return name, test["Source_Excel_Row"].map(preds).to_numpy()


def main(args):
    folder = Path(args.worksheet_dir)
    master = pd.read_excel(MASTER_FILE, sheet_name=MASTER_SHEET)
    test = load(folder / "Annotation_Worksheet.xlsx", master, list(ZERO_SHOT.values()))
    if test.empty:
        stop("Annotation_Worksheet.xlsx has no labels yet - it is the test set, so label it first")
    train_path = folder / "Training_Worksheet.xlsx"
    train = load(train_path, master, []) if train_path.exists() else pd.DataFrame(columns=["Feedback_Text", "Human_Label", "Source_Excel_Row"])
    if set(train["Source_Excel_Row"]) & set(test["Source_Excel_Row"]):
        stop("training and test worksheets share rows")
    y_test = test["Human_Label"].astype(str).to_numpy()
    x_test, x_train, y_train = test["Feedback_Text"], train["Feedback_Text"], train["Human_Label"].astype(str)
    print(f"Test set: {len(test)} labelled rows; training set: {len(train)} labelled rows")

    cache_path = folder / "model_predictions.json"
    fingerprint = hashlib.sha1(json.dumps([list(map(int, test["Source_Excel_Row"])), list(map(int, train["Source_Excel_Row"])),
                                           list(train["Human_Label"].astype(str))]).encode()).hexdigest()
    cache = {} if args.fresh else load_cache(cache_path, fingerprint)
    runs, kinds, unparsed = {}, {}, {}

    def method(name, kind, compute):
        if name in cache:
            print(f"  {name}: reused from {cache_path.name}")
        else:
            result = compute()
            cache[name] = {"kind": kind, "runs": [list(map(str, r)) for r in result["runs"]], "unparsed": result.get("unparsed")}
            save_cache(cache_path, fingerprint, cache)
        runs[name] = [np.array(r) for r in cache[name]["runs"]]
        kinds[name] = cache[name]["kind"]
        if cache[name].get("unparsed") is not None:
            unparsed[name] = cache[name]["unparsed"]

    for name, col in ZERO_SHOT.items():
        runs[name], kinds[name] = [test[col].fillna("Neutral").astype(str).to_numpy()], "zero-shot baseline"

    has_train = len(train) >= 30 and y_train.nunique() == len(LABELS)
    if not has_train:
        print("Training worksheet has fewer than 30 labels or misses a class - trained models skipped")
    else:
        method("TF-IDF logistic regression", "trained baseline", lambda: {"runs": [mc.tfidf_logreg(x_train, y_train, x_test)]})
        if not args.skip_setfit:
            name = f"SetFit ({args.setfit_backbone.split('/')[-1]})"
            method(name, "trained (few-label)", lambda: {"runs": [timed(f"{name} seed {s}", lambda s=s: mc.setfit_predict(x_train, y_train, x_test, s, args.setfit_backbone)) for s in range(args.seeds)]})
        for backbone in ([] if args.skip_finetune else args.finetune_backbone):
            name = f"Fine-tuned {backbone.split('/')[-1]}"
            method(name, "trained (fine-tuned)", lambda: {"runs": [timed(f"{name} seed {s}", lambda s=s: mc.finetune_predict(x_train, y_train, x_test, s, backbone, epochs=args.epochs)) for s in range(args.seeds)]})

    for model_name in args.llm:
        short = model_name.split("/")[-1]
        settings = [("zero-shot", pd.DataFrame(columns=["text", "label"]))]
        if has_train:
            settings.append((f"{args.shots}-shot", mc.few_shot_examples(x_train, y_train, args.shots, seed=42)))
        for setting, examples in settings:
            name = f"LLM {short} ({setting})"
            def run_llm(name=name, examples=examples):
                pred, bad = timed(name, lambda: mc.llm_predict(x_test, model_name, examples, args.llm_4bit, args.llm_dtype))
                return {"runs": [pred], "unparsed": bad}
            try:
                method(name, "LLM (prompted)", run_llm)
            except Exception as error:  # a model that cannot load on this machine is reported, not fatal
                print(f"  {name}: skipped ({type(error).__name__}: {str(error)[:120]})")
                import gc, torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    for spec in args.external:
        name, pred = external_predictions(spec, test)
        runs[name], kinds[name] = [pred], "LLM (prompted, external)"

    rows, final = [], {}
    for name, method_runs in runs.items():
        if name.startswith("__"):
            continue
        row, final[name] = mc.method_row(name, kinds[name], y_test, method_runs)
        if name in unparsed:
            row["unparsed_replies"] = unparsed[name]
        rows.append(row)
    methods = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print(methods[["method", "accuracy", "macro_f1", "macro_f1_ci95"]].to_string(index=False))

    best = methods.iloc[0]["method"]
    significance = pd.DataFrame([
        {"best_method": best, "compared_with": name, "best_right_other_wrong": b, "other_right_best_wrong": c,
         "mcnemar_exact_p": round(p, 4), "significant_at_0.05": p < 0.05}
        for name in methods["method"][1:] for b, c, p in [mc.mcnemar(y_test, final[best], final[name])]
    ])

    errors = test[["Source_Excel_Row", "Scheme", "Feedback_Text", "Human_Label"]].copy()
    for name, pred in final.items():
        errors[name] = pred
    errors["methods_wrong"] = sum((errors[name] != errors["Human_Label"]).astype(int) for name in final)
    errors = errors.sort_values("methods_wrong", ascending=False)

    summary_rows = [
        {"item": "Test set", "value": f"{len(test)} human-labelled rows from Annotation_Worksheet.xlsx (never used for training)"},
        {"item": "Training set", "value": f"{len(train)} human-labelled rows from Training_Worksheet.xlsx"},
        {"item": "Test label distribution", "value": ", ".join(f"{k} {v}" for k, v in pd.Series(y_test).value_counts().items())},
        {"item": "Seeds per trained model", "value": f"{args.seeds} (table uses the majority vote across seeds; mean ± SD shown separately)"},
        {"item": "Confidence intervals", "value": "95% bootstrap, 1,000 resamples"},
        {"item": "Significance", "value": "Exact McNemar test, best method vs each other method"},
        {"item": "Best method", "value": best},
    ]
    if "AI_Suggestion" in train.columns and len(train):
        changed = int((train["Human_Label"] != train["AI_Suggestion"]).sum())
        summary_rows.append({"item": "Training labels", "value": f"LLM-assisted, human-verified: the annotator changed {changed} of {len(train)} "
                             f"AI suggestions ({100 * changed / len(train):.1f}%); test labels are human-only"})
    rater2 = folder / "Annotation_Worksheet_rater2.xlsx"
    if rater2.exists():
        agreement = inter_annotator_kappa(read_worksheet(folder / "Annotation_Worksheet.xlsx", ["Human_Label"])[0],
                                          read_worksheet(rater2, ["Human_Label"])[0], "Human_Label")
        if agreement:
            summary_rows.append({"item": "Human agreement (ceiling)", "value": f"kappa {agreement['cohen_kappa']} on {agreement['n']} rows ({agreement['percent_agreement']}% raw)"})

    sheets = {
        "Summary": pd.DataFrame(summary_rows), "Methods": methods, "Significance": significance,
        "Per Class": pd.concat([per_class(y_test, p).assign(method=n) for n, p in final.items()], ignore_index=True),
        "Confusion": pd.concat([confusion(y_test, p).assign(method=n) for n, p in final.items()], ignore_index=True),
        "Error Analysis": errors,
    }
    if args.llm:
        sheets["LLM Prompt"] = pd.DataFrame({"prompt": mc.LLM_INSTRUCTIONS.split("\n") + ["", f"Few-shot: {args.shots} training examples per class, seed 42; greedy decoding"]})
    if has_train and not args.no_learning_curve:
        if "__learning_curve__" not in cache:
            cache["__learning_curve__"] = timed("learning curve", lambda: learning_curve(args, train, test, y_test)).to_dict("records")
            save_cache(cache_path, fingerprint, cache)
        sheets["Learning Curve"] = pd.DataFrame(cache["__learning_curve__"])
    absa_path = folder / "ABSA_Annotation_Worksheet.xlsx"
    if not args.skip_absa and absa_path.exists():
        absa = absa_comparison(absa_path, master, args.absa_model)
        if absa is not None:
            sheets["ABSA Comparison"] = absa
            print(absa[["method", "detected", "aspect_mentions", "macro_f1"]].to_string(index=False))

    results = folder / "model_comparison_results.xlsx"
    save_workbook(results, sheets)
    print(f"Saved {results}")


if __name__ == "__main__":
    class HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    parser.add_argument("--seeds", type=int, default=5, help="training runs per trained model")
    parser.add_argument("--epochs", type=int, default=4, help="fine-tuning epochs")
    parser.add_argument("--setfit-backbone", default="sentence-transformers/all-mpnet-base-v2", help="SetFit sentence encoder")
    parser.add_argument("--finetune-backbone", action="append", default=[], help="transformer to fine-tune; repeatable (default: microsoft/deberta-v3-base)")
    parser.add_argument("--llm", action="append", default=[], help="Hugging Face model id; repeatable (needs a GPU in practice)")
    parser.add_argument("--shots", type=int, default=3, help="few-shot examples per class for LLMs")
    parser.add_argument("--llm-dtype", default="auto", choices=["auto", "float32", "bfloat16"], help="LLM weights precision (bfloat16 halves CPU memory)")
    parser.add_argument("--external", action="append", default=[], help='precomputed predictions, "Name=path.csv" with Source_Excel_Row,label; repeatable')
    parser.add_argument("--fresh", action="store_true", help="ignore cached predictions from earlier runs")
    parser.add_argument("--llm-4bit", action="store_true", help="load LLMs in 4-bit (needs bitsandbytes and a CUDA GPU)")
    parser.add_argument("--absa-model", default="yangheng/deberta-v3-base-absa-v1.1", help="aspect-sentiment model")
    parser.add_argument("--skip-setfit", action="store_true")
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--skip-absa", action="store_true")
    parser.add_argument("--no-learning-curve", action="store_true")
    parser.add_argument("--curve-setfit", action="store_true", help="include SetFit in the learning curve (slow on CPU)")
    parser.add_argument("--worksheet-dir", default=str(OUTPUT_DIR), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.finetune_backbone = args.finetune_backbone or ["microsoft/deberta-v3-base"]
    main(args)
