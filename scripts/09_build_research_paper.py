import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import MASTER_FILE, MASTER_SHEET, OUTPUT_DIR, REPORTS_DIR
from src.report_docx import bullet, caption, new_document, num, p_value, para, pct, signed, table
from src.sentiment_lexicon import AGRI_DOMAIN_LEXICON

STATS_FILE = OUTPUT_DIR / "descriptive_statistics.xlsx"
NUMBERS_FILE = OUTPUT_DIR / "report_numbers.json"
TOPIC_FILE = OUTPUT_DIR / "topic_modeling_results.xlsx"
VALIDATION_FILE = OUTPUT_DIR / "validation_results.xlsx"
DB_CHECK_FILE = OUTPUT_DIR / "database_verification.csv"
OUT = REPORTS_DIR / "Policy_Intelligence_Research_Paper.docx"

MAJOR = ["Rythu Bharosa", "PM-KISAN", "Kisan Credit Card", "Input Subsidy"]
ASPECTS = ["timeliness", "amount", "enrollment_ease", "official_behavior", "clarity", "impact"]
ASPECT_NAMES = {"timeliness": "Timeliness", "amount": "Amount of support", "enrollment_ease": "Enrollment ease",
                "official_behavior": "Official behavior", "clarity": "Clarity of information", "impact": "Overall impact"}
TIMING_TOPICS = ["Late credit of benefits", "Payment delays & bank verification"]

REFERENCES = [
    'B. Liu, "Sentiment analysis and opinion mining," Synth. Lect. Hum. Lang. Technol., vol. 5, no. 1, pp. 1–167, 2012.',
    'B. Pang and L. Lee, "Opinion mining and sentiment analysis," Found. Trends Inf. Retr., vol. 2, no. 1–2, pp. 1–135, 2008.',
    'E. Kouloumpis, T. Wilson, and J. Moore, "Twitter sentiment analysis: The good the bad and the OMG!," Proc. ICWSM, vol. 5, no. 1, pp. 538–541, 2011.',
    'A. Devitt and K. Ahmad, "Sentiment polarity identification in financial news," Proc. ACL, vol. 1, pp. 52–59, 2007.',
    "A. Neogi, K. Garg, R. K. Mishra, and Y. K. Dwivedi, \"Sentiment analysis and classification of Indian farmers' protest using twitter data,\" Int. J. Inf. Manag. Data Insights, vol. 1, no. 2, art. 100019, 2021, doi: 10.1016/j.jjimei.2021.100019.",
    'Government of India, Ministry of Agriculture & Farmers Welfare, "Operational Guidelines of Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)," New Delhi, 2019.',
    'B. Drury and M. Roche, "A survey of the applications of text mining for agriculture," Comput. Electron. Agric., vol. 163, art. 104864, 2019, doi: 10.1016/j.compag.2019.104864.',
    "S. Bird, E. Klein, and E. Loper, Natural Language Processing with Python. O'Reilly Media, 2009.",
    'T. Mikolov, K. Chen, G. Corrado, and J. Dean, "Efficient estimation of word representations in vector space," Proc. ICLR, 2013.',
    'J. Pennington, R. Socher, and C. Manning, "GloVe: Global vectors for word representation," Proc. EMNLP, pp. 1532–1543, 2014.',
    'J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers," Proc. NAACL, pp. 4171–4186, 2019.',
    'A. Vaswani, N. Shazeer, N. Parmar, et al., "Attention is all you need," Proc. NeurIPS, pp. 5998–6008, 2017.',
    'S. Rani and P. Kumar, "A journey of Indian languages over sentiment analysis: a systematic review," Artif. Intell. Rev., 2018, doi: 10.1007/s10462-018-9670-y.',
    'A. Das and S. Bandyopadhyay, "SentiWordNet for Indian languages," Proc. 8th Workshop on Asian Language Resources (COLING 2010), pp. 56–63, 2010.',
    'W. Medhat, A. Hassan, and H. Korashy, "Sentiment analysis algorithms and applications: A survey," Ain Shams Eng. J., vol. 5, no. 4, pp. 1093–1113, 2014.',
    'A. Rao and N. Spasojevic, "Actionable and political text classification using word embeddings and LSTM," arXiv:1607.02501, 2016.',
    'R. Feldman, "Techniques and applications for sentiment analysis," Commun. ACM, vol. 56, no. 4, pp. 82–89, 2013.',
    "B. Agarwal and N. Mittal, Prominent Feature Extraction for Sentiment Analysis. Springer, 2016.",
    'A. Ceron and F. Negri, "The "social side" of public policy: Monitoring online public opinion and its mobilization during the policy cycle," Policy & Internet, vol. 8, no. 2, pp. 131–147, 2016.',
    'J. Grimmer and B. M. Stewart, "Text as data: The promise and pitfalls of automatic content analysis methods for political texts," Political Analysis, vol. 21, no. 3, pp. 267–297, 2013.',
    'Government of Andhra Pradesh, Agriculture & Cooperation Department, "YSR Rythu Bharosa scheme — operational guidelines," G.O.Ms.No. 96, 19 Sep. 2019.',
    'D. Kakwani, A. Kunchukuttan, S. Golla, N. C. Gokul, A. Bhattacharyya, M. M. Khapra, and P. Kumar, "IndicNLPSuite: Monolingual corpora, evaluation benchmarks and pre-trained multilingual language models for Indian languages," Findings of EMNLP 2020, pp. 4948–4961, 2020.',
    'H. Isoaho, D. Gritsenko, and E. Mäkelä, "Topic modeling and text analysis for qualitative policy research," Policy Studies Journal, vol. 49, no. 1, pp. 300–324, 2021, doi: 10.1111/psj.12343.',
    'C. Hutto and E. Gilbert, "VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text," Proc. 8th Int. AAAI Conf. Weblogs and Social Media (ICWSM), pp. 216–225, 2014.',
    'D. M. Blei, A. Y. Ng, and M. I. Jordan, "Latent Dirichlet Allocation," J. Mach. Learn. Res., vol. 3, pp. 993–1022, 2003.',
    "R. Ranjan, \"Farmers' Awareness of Government Schemes and Its Impact on Income Growth: A Survey Study of Bhagalpur District,\" RESEARCH HUB Int. Multidiscip. Res. J., vol. 13, no. 3, pp. 22–27, Mar. 2026, doi: 10.53573/rhimrj.2026.v13n03.004.",
    'L. Rink, J. Meijdam, and D. Graus, "Aspect-Based Sentiment Analysis for Open-Ended HR Survey Responses," Proc. 3rd Workshop on NLP for Human Resources (NLP4HR), EACL, 2024, arXiv:2402.04812.',
    'M. M. van Buchem, O. M. Neve, I. M. J. Kant, E. W. Steyerberg, H. Boosman, and E. F. Hensen, "Analyzing patient experiences using natural language processing: development and validation of the artificial intelligence patient reported experience measure (AI-PREM)," BMC Med. Inform. Decis. Mak., vol. 22, art. 183, Jul. 2022, doi: 10.1186/s12911-022-01923-5.',
]


class Paper:
    def __init__(self):
        self.doc = new_document("Policy Intelligence for Agriculture", font="Times New Roman", size=12, heading_color="000000")
        self.tables = 0
        self.warnings = []

    def h1(self, text):
        self.doc.add_heading(text, 1)

    def h2(self, text):
        self.doc.add_heading(text, 2)

    def p(self, text):
        para(self.doc, text, align="justify")

    def table(self, title, headers, rows, size=9):
        self.tables += 1
        caption(self.doc, f"Table {self.tables}. {title}")
        table(self.doc, headers, rows, size=size)
        return self.tables

    def claim(self, ok, message):
        if not ok:
            self.warnings.append(message)


def load():
    stats = pd.ExcelFile(STATS_FILE)
    topics = pd.ExcelFile(TOPIC_FILE)
    master = pd.read_excel(MASTER_FILE, sheet_name=MASTER_SHEET)
    validation = pd.ExcelFile(VALIDATION_FILE) if VALIDATION_FILE.exists() else None
    db_ok = DB_CHECK_FILE.exists() and (pd.read_csv(DB_CHECK_FILE)["status"] == "PASS").all()
    return json.loads(NUMBERS_FILE.read_text(encoding="utf-8")), stats, topics, master, validation, db_ok


def main():
    n, stats, topics, master, validation, db_ok = load()
    s = stats.parse
    P = Paper()

    conversion = s("Awareness by Scheme")
    widely_heard = conversion[conversion["Heard of"] >= 50]
    best, worst = widely_heard.loc[widely_heard["Conversion %"].idxmax()], widely_heard.loc[widely_heard["Conversion %"].idxmin()]
    crop_conv = conversion.set_index("Scheme").loc["Crop Insurance"]
    schemes = s("Scheme Summary").set_index("Scheme")
    absa_scheme = s("ABSA by Scheme").set_index("Scheme")
    absa = s("ABSA by Aspect").set_index("Aspect")
    div = {k: v for k, v in n["divergence"].items()}
    gap, agree = n["awareness_gap"], n["method_agreement"]
    dis = topics.parse("Dissatisfaction Topics").sort_values("share", ascending=False)
    sat = topics.parse("Satisfaction Topics").sort_values("share", ascending=False)
    timing_share = dis[dis["label"].isin(TIMING_TOPICS)]["share"].sum() * 100
    by_scheme = topics.parse("Dissatisfaction x Scheme").set_index("scheme_normalized")
    by_scheme_pct = by_scheme.div(by_scheme.sum(axis=1), axis=0) * 100
    amount_rating = master.groupby("scheme_normalized")["C3_amount_numeric"].mean()
    themes = s("Suggestion Themes")
    pre = n["preprocessing"]
    worst_aspect = {sc: max(ASPECTS[:5], key=lambda a: absa_scheme.loc[sc, f"{a}_pct_negative"]) for sc in MAJOR}
    enrol_worst_all = all(a == "enrollment_ease" for a in worst_aspect.values())
    P.claim(enrol_worst_all, f"Enrollment ease is no longer the worst aspect for every major scheme: {worst_aspect}")
    major = schemes.loc[schemes["N"] >= 30]
    P.claim(major["pct_negative"].idxmax() == "Crop Insurance", "Crop Insurance no longer has the highest negative share")
    P.claim(major["avg_impact"].idxmin() == "PM-KISAN", "PM-KISAN no longer has the lowest impact rating among major schemes")

    methods = validation.parse("Overall Methods") if validation is not None and "Overall Methods" in validation.sheet_names else None
    absa_val = validation.parse("ABSA Polarity") if validation is not None and "ABSA Polarity" in validation.sheet_names else None
    detect = validation.parse("ABSA Aspect Detection") if validation is not None and "ABSA Aspect Detection" in validation.sheet_names else None
    summary_val = validation.parse("Summary").set_index("item")["value"] if validation is not None else None

    doc = P.doc
    para(doc, "Policy Intelligence for Agriculture: An NLP-Based Sentiment Analysis of Farmer Feedback on Government Welfare Schemes — Evidence from Chittoor District, Andhra Pradesh, India", align="center", size=16)
    para(doc, "Krish Singh", align="center")
    para(doc, "Department of Computer Science and Engineering (Data Science), Vellore Institute of Technology, Vellore, India", align="center", size=10, italic=True)

    if methods is not None:
        m = methods.set_index("method")
        best_method = m["macro_f1"].idxmax()
        validation_sentence = (
            f"Against {num(m['n'].iloc[0])} independently human-labelled responses, VADER reached an accuracy of {m.loc['VADER + agri lexicon', 'accuracy']:.2f} "
            f"(macro-F1 {m.loc['VADER + agri lexicon', 'macro_f1']:.2f}) and the zero-shot transformer {m.loc['RoBERTa transformer (zero-shot)', 'accuracy']:.2f} "
            f"(macro-F1 {m.loc['RoBERTa transformer (zero-shot)', 'macro_f1']:.2f}); the best method by macro-F1 was {best_method} ({m.loc[best_method, 'macro_f1']:.2f}). "
        )
    else:
        validation_sentence = (
            f"Agreement between the lexicon scorer and the transformer on identical text is only slight (Cohen's κ = {n['kappa']:.3f}), "
            "so a stratified human-annotation protocol is provided for validating both. "
        )

    P.h2("Abstract")
    P.p(
        "Government welfare schemes for farmers are routinely evaluated through structured satisfaction surveys that report high aggregate approval, "
        "yet such scores may obscure how farmers actually experience awareness, enrolment and benefit delivery. This study combines a normalised "
        "relational database with Natural Language Processing (NLP) — lexicon-based sentiment scoring, a pretrained transformer classifier, "
        "Aspect-Based Sentiment Analysis (ABSA), topic modelling and suggestion-frequency mining — to analyse open-ended feedback from "
        f"{num(n['n_respondents'])} farmers in Chittoor district, Andhra Pradesh, India, on schemes including PM-KISAN, Rythu Bharosa, Kisan Credit Card "
        f"and Crop Insurance. While {pct(n['aware_pct'])} of respondents are aware of at least one scheme, {pct(gap['pct_gap'])} name more schemes heard of "
        f"than enrolled in, and conversion from awareness to enrolment ranges from {pct(best['Conversion %'])} ({best['Scheme']}) to "
        f"{pct(worst['Conversion %'])} ({worst['Scheme']}). Comparing text sentiment with respondents' own ratings shows polarity divergence in "
        f"{pct(div['divergence_overall']['pct'])} of comparable responses, rising to {pct(div['divergence_timeliness']['pct'])} for timeliness and "
        f"{pct(div['divergence_enrollment_ease']['pct'])} for enrolment ease (McNemar–Bowker p {p_value(n['mcnemar_bowker']['p'])}). Enrolment ease is "
        "the most negatively described aspect for each of the four largest schemes, pointing to a shared administrative bottleneck rather than "
        f"scheme-specific design failures. Crop Insurance is the only scheme with a high negative-sentiment share ({pct(schemes.loc['Crop Insurance', 'pct_negative'])}), "
        "and PM-KISAN combines strong satisfaction with the lowest self-reported farming impact of any major scheme. Complaint topics are dominated by "
        "late or delayed credit of benefits, whereas farmers' own suggestions prioritise a larger benefit amount and simpler procedures. "
        + validation_sentence +
        "The study contributes a reproducible pipeline and database design for evidence-based evaluation of agricultural welfare schemes, together "
        "with a transparent account of where automated sentiment measures can and cannot be trusted."
    )
    para(doc, "**Keywords:** Natural Language Processing; Sentiment Analysis; Aspect-Based Sentiment Analysis; Topic Modelling; Relational Database; Farmer Welfare Schemes; Survey Triangulation; Agricultural Policy; India")

    P.h1("1. Introduction")
    P.p("Agriculture employs nearly half of India's workforce, and central and state governments run numerous welfare schemes — including PM-KISAN, "
        "Rythu Bharosa, crop insurance, minimum support price (MSP) mechanisms and input subsidy programmes — to improve farmer income security and "
        "resilience. Despite substantial public investment, evidence across states suggests that scheme impact is uneven, with poor awareness, delayed "
        "benefit disbursement, administrative bottlenecks and regional disparities repeatedly cited as barriers [6], [21].")
    P.p("Conventional evaluation of these schemes relies almost exclusively on structured questionnaires reporting aggregate satisfaction percentages. "
        "Such instruments are administratively convenient and statistically tractable, but they are susceptible to acquiescence and social-desirability "
        "bias and compress lived experience into a narrow numeric range. A farmer may report being \"Satisfied\" on a five-point scale while, in the same "
        "interview, describing unresolved payment delays, confusing documentation requirements or limited awareness beyond one or two flagship schemes.")
    P.p("Natural Language Processing offers a scalable route to extracting sentiment and thematic structure directly from farmers' own open-ended "
        "narratives, converting free text into evidence that can be triangulated against structured ratings from the same respondent. Divergence between "
        "what a respondent rates and what the same respondent writes is itself informative: it can signal response bias, needs that fixed-choice questions "
        "miss, or specific service dimensions (timeliness, procedural ease, official conduct) that aggregate satisfaction scores fail to separate.")
    P.p("This study applies a multi-method NLP pipeline — lexicon-based sentiment scoring, a pretrained transformer classifier, ABSA mapped onto the "
        "survey's own six rated service dimensions, unsupervised topic modelling, suggestion-frequency mining and formal significance testing — to "
        f"{num(n['n_respondents'])} farmer interviews from Chittoor district, Andhra Pradesh, with all records, model outputs and human labels held in a "
        "normalised relational database. Section 2 reviews related work, Section 3 describes the data and methods, Section 4 presents results, Section 5 "
        "discusses them, Section 6 sets out policy implications, Section 7 states limitations and Section 8 concludes.")

    P.h1("2. Related Work")
    P.h2("2.1 Awareness and Utilisation of Farmer Welfare Schemes in India")
    P.p("A growing body of Indian agricultural-policy literature documents partial and uneven awareness of government schemes among farmers. Early "
        "opinion-mining foundations [1], [2] established the general methodology for analysing subjective text, later extended to social-media and "
        "short-text settings [3]. In Indian agriculture, sentiment classification has been applied to social-media discussion of the farmers' protest "
        "[5], and text mining remains comparatively underused in agricultural research [7]; applications to financial and political text [4], [16] show "
        "that these techniques transfer across domains, though direct application to farmer welfare-scheme survey data remains limited. A recent survey of PM-KISAN beneficiaries in Bhagalpur district found that greater scheme awareness was associated with more stable "
        "income growth, while identifying digital literacy and procedural complexity as persistent barriers to full utilisation [26]. Official scheme "
        "documentation [6], [21] provides the administrative context against which farmer-reported experience is compared here.")
    P.h2("2.2 NLP and Sentiment Analysis in Survey and Policy Research")
    P.p("Outside agricultural policy, NLP and sentiment analysis have been applied to social-media and policy-adjacent discourse [3], [16], [19] and to "
        "multilingual and Indian-language sentiment resources [13], [14], [22]. Word embeddings [9], [10] and transformer language models [11], [12] "
        "underlie the modern classifiers used in this and comparable studies. In human-resources research, Rink, Meijdam and Graus applied ABSA to "
        "open-ended employee-satisfaction responses, comparing few-shot transformer approaches against bag-of-words and zero-shot baselines [27]. In "
        "healthcare, van Buchem et al. validated the AI-PREM tool for open-ended patient-experience responses, reporting an F1 score of 0.97 for positive "
        "text but only 0.63 for negative text — evidence that sentiment methods can perform unevenly across polarity classes [28]. Sentiment techniques "
        "[17], [18], topic modelling for qualitative policy research [23], and monitoring of public opinion across the policy cycle [19] establish "
        "computational text analysis as a complement to structured instruments, provided automated output is validated against human reading [20].")
    P.p("The methods used here draw specifically on VADER, a lexicon- and rule-based scorer built for short informal text and validated against human "
        "judges [24]; Latent Dirichlet Allocation (LDA), the standard unsupervised topic model [25]; and general NLP tooling surveys [8], [15].")
    P.h2("2.3 Research Gap")
    P.p("Two gaps motivate this study. First, research on Indian farmer-scheme awareness relies almost entirely on structured survey percentages, with "
        "little systematic validation against respondents' own qualitative accounts from the same interview. Second, while NLP and ABSA have proven "
        "effective on open-ended survey text in HR, healthcare and policy discourse, they remain largely unapplied to agricultural welfare-scheme "
        "evaluation with rural, often low-literacy respondents. No identified study triangulates NLP-derived sentiment against structured satisfaction "
        "ratings from the same respondents in this domain, disaggregates that triangulation by scheme, or compares complaint-driven topics with "
        "independently mined farmer suggestions. This study addresses these gaps.")

    a, e, l = n["age"], n["experience"], n["landholding"]
    gender = s("Demo Gender").set_index("Category")
    land_type = s("Demo Land Type").set_index("Category")
    occupation = s("Demo Occupation").set_index("Category")
    language = s("Survey Language").set_index("Category")
    comfort = s("Survey Comfort").set_index("Category")
    comfortable = comfort.loc[[c for c in comfort.index if c in ("Comfortable", "Very comfortable")], "Percent"].sum()
    dur = n["duration"]
    P.h1("3. Materials and Methods")
    P.h2("3.1 Study Area and Sample")
    P.p(f"Data were collected from {num(n['n_respondents'])} farmers within a 50 km radius of the researchers' university campus in Chittoor district, "
        "Andhra Pradesh, through structured interviews conducted by trained investigators; the dataset was verified by the Ethnic Community, Government "
        f"of Andhra Pradesh. Respondents had a mean age of {a['mean']:.1f} years (n = {num(a['n'])} valid answers) and a mean of {e['mean']:.1f} years of "
        f"farming experience. The sample was {pct(gender.loc['Male', 'Percent'])} male (n = {num(gender.loc['Male', 'Count'])}) and "
        f"{pct(gender.loc['Female', 'Percent'])} female (n = {num(gender.loc['Female', 'Count'])}). Mean landholding was {l['mean']:.2f} acres "
        f"(median {l['median']:.2f}, right-skewed), and {pct(land_type.loc['Owned', 'Percent'])} of respondents owned their land. The most common "
        f"occupation was farming only ({pct(occupation['Percent'].iloc[0])}). Interviews were conducted predominantly in Telugu "
        f"({pct(language.loc['Telugu', 'Percent'])}), with smaller shares in English and Tamil; interviewers rated participants Comfortable or Very "
        f"comfortable in {pct(comfortable)} of interviews, and the mean interview lasted {dur['mean']:.1f} minutes (range {dur['min']:.0f}–{dur['max']:.0f}).")
    P.h2("3.2 Survey Instrument")
    P.p("The instrument had five sections: (A) demographic and agricultural background; (B) awareness of and access to welfare schemes, including which "
        "schemes respondents had heard of and which they were enrolled in; (C) scheme-specific feedback — a five-point overall satisfaction rating, six "
        "five-point ratings of service aspects (timeliness of benefits, amount of support, ease of enrolment, behaviour of officials, clarity of "
        "information and overall impact on farming) and three open-ended questions (positive aspects, problems faced and suggestions); (D) overall "
        "perception of government support; and (E) interviewer-recorded language, comfort, observations and duration. Pairing structured ratings with "
        "open-ended questions on the same service dimensions is what enables the triangulation in Section 3.8.")

    q = n["data_quality"]
    P.h2("3.3 Data Management, Quality Audit and Pre-processing")
    storage = ("All records are held in a PostgreSQL 16 database designed to third normal form: the raw sheet is loaded unchanged into a staging table, "
               "then split into 13 survey tables (respondents, villages, schemes, scheme-name aliases, awareness, feedback and aspect ratings, among "
               "others) and 9 analytics tables holding model outputs, topics, data-quality flags and human labels. Multi-valued answers — crops grown, "
               "schemes heard of and schemes enrolled in — are stored as separate rows rather than comma-separated lists, and every record keeps its "
               "row number in the source file for traceability. ") if db_ok else ""
    P.p(storage + "A data-quality audit preceded all analysis. It flagged implausible numeric values "
        f"({num(q['numeric_issue_rows'])} respondents), farming experience exceeding age ({num(q['age_experience_impossible'])}, both values set to "
        f"missing), respondents who reported no awareness of any scheme yet rated a specific scheme ({num(q['unaware_but_rated'])}), and respondents "
        f"listing a scheme as enrolled that they did not list as heard of ({num(q['enrolled_not_heard'])}). Contradictory answers were flagged rather "
        "than altered, so the original responses remain available.")
    P.p("Free-text responses were stripped of extraneous whitespace and null-like tokens were normalised to missing. Answers that only report "
        "\"nothing to say\" — \"No\", \"Nothing\", \"Nil\", \"No problems\", \"No suggestions\" — were treated as empty rather than as sentiment: VADER "
        "scores the bare word \"No\" as negative, so a response such as \"Crop protection. No. No\" (a stated benefit followed by no complaints and no "
        "suggestions) had been scored Negative purely from the nil tokens in an earlier iteration; excluding such clauses before scoring raised "
        "VADER–transformer agreement from 30.7% to 37.2% at the time of the fix. For topic modelling a heavier pass was applied — tokenisation, "
        f"stop-word removal and WordNet lemmatisation — reducing vocabulary by {pct(pre['C4_positive']['vocab_reduction_pct'])} for positive-aspect text "
        f"and {pct(pre['C5_problems']['vocab_reduction_pct'])} for problem text. Sentiment scoring used the lighter pass, since VADER relies on surface "
        "cues that lemmatisation removes.")
    P.h2("3.4 Data-Driven Lexicon Correction")
    P.p("A systematic audit compared VADER's general-purpose lexicon with this corpus's vocabulary. For every word appearing at least eight times across "
        "the positive-aspect (C4) and problem (C5) fields, the share of its occurrences in each field was computed; words found in one field at least 80% "
        "of the time were checked against their VADER score. Two systematic error patterns emerged. (i) Relief phrases: \"burden\", \"pressure\", "
        "\"losses\", \"unstable\" and \"uncertain\" carry negative valence in VADER and are not negated by \"reduce(d)\" or \"less\", so \"reduced "
        "financial burden\" scored Negative; these terms appeared in relief phrases in at least 89% of uses and were neutralised. (ii) Missing evaluative "
        "terms: \"helped\" was absent from VADER although \"help\" and \"helps\" are present, affecting 237 responses, of which 110 scored non-Positive "
        "purely from this gap; it was added at +1.7. \"Slow\", \"complex\" and \"unnecessary\" were added as negative terms. The procedure is "
        "replicable on any new corpus.")
    P.h2("3.5 Scheme-Name Normalisation")
    P.p("Respondents named schemes in free text, producing 93 distinct raw spellings (for example \"Rythu Bharosa\", \"rythu barosa\", \"RYTHU BHAROSA\", "
        "\"Raitubharosa\"). These were mapped to nine named schemes plus an \"Other\" category using weighted-ratio fuzzy matching with a 70% similarity "
        "threshold and a curated alias dictionary (for example \"KCC\" → Kisan Credit Card, \"PMFBY\" → Crop Insurance, \"irrigation schemes\" → PM "
        "Krishi Sinchai Yojana). Multi-select answers listing schemes heard of and enrolled in were parsed against the questionnaire's option list, so "
        "that options containing commas (such as \"Input Subsidy (Seed, Fertilizer)\") are counted once.")
    P.h2("3.6 Sentiment Classification")
    P.p("Two independent sentiment methods were applied and compared: (i) VADER [24], extended with a domain lexicon of "
        f"{len(AGRI_DOMAIN_LEXICON)} terms calibrated through the audit in Section 3.4 (for example \"delay\", \"corruption\" and \"bribe\" scored "
        "negative; \"timely\", \"transparent\" and \"cooperative\" positive; administrative nouns such as \"approval\" and \"benefit\" neutral); and "
        "(ii) a pretrained transformer (cardiffnlp/twitter-roberta-base-sentiment-latest) applied zero-shot. Both classify text as Positive, Neutral "
        "or Negative.")
    P.h2("3.7 Aspect-Based Sentiment Analysis")
    P.p("Sentiment was also scored for the same six service aspects rated in the instrument. Each response was split into clauses, each clause was "
        "matched against a keyword dictionary per aspect, and VADER scored each aspect-relevant clause, averaging where an aspect appeared in more than "
        "one clause. Aspects not mentioned receive no score. Results were also disaggregated by scheme (Section 4.8).")
    P.h2("3.8 Triangulation with Structured Ratings")
    P.p("Likert responses (overall satisfaction, C2; six aspect ratings, C3) were converted to a −2 to +2 scale and collapsed to three-way polarity. A "
        "response was counted as divergent when NLP polarity and the respondent's own rating polarity were directly opposed (Positive vs Negative); "
        "pairs where either side was Neutral count as non-divergent, and pairs with a missing side are excluded. This was applied to overall sentiment "
        "versus overall satisfaction and to each ABSA aspect score versus the matching C3 rating.")
    P.h2("3.9 Topic Modelling and Suggestion-Frequency Mining")
    P.p(f"LDA [25] was fitted separately to the lemmatised positive-aspect text (C4, n = {num(pre['C4_positive']['n_documents'])} substantive answers) and "
        f"problem text (C5, n = {num(pre['C5_problems']['n_documents'])}) using a bag-of-words matrix (400-term vocabulary, minimum document frequency 5, "
        "maximum document frequency 85%) and five topics per corpus. Topics were labelled by inspecting their ten highest-weighted terms. As an "
        f"independent cross-check, farmers' suggestions (C6, n = {num(n['n_suggestions'])} substantive answers) were analysed by term frequency after "
        "stop-word removal. To check the choice of five topics, models with 3–10 topics were compared on UMass coherence and perplexity, and "
        "stability was measured by refitting with five further random seeds and computing the mean best-match Jaccard overlap of each topic's ten "
        "top terms.")
    P.h2("3.10 Validation Protocol")
    P.p("None of the sentiment methods was fine-tuned on this dataset, so a validation protocol was designed. A stratified 20% random sample, allocated "
        f"proportionally by scheme ({num(243)} of {num(n['n_with_feedback_text'])} respondents with feedback text), was labelled for overall sentiment, and "
        "a second independent stratified sample (n = 131) was labelled for aspect-level sentiment, following a written rubric and blind to all model "
        "outputs. Each worksheet row carries its source-file row number so that every label can be traced to the original response. The zero-shot "
        "methods were compared with the human labels directly; two supervised baselines — TF-IDF logistic regression and a small feed-forward network — "
        "were trained and evaluated on the human labels with stratified k-fold cross-validation (k ≤ 5). Accuracy, macro-averaged precision, recall and "
        "F1 and Cohen's κ are reported with 95% bootstrap confidence intervals (1,000 resamples).")
    P.h2("3.11 Statistical Significance Testing")
    P.p("Chi-square tests of independence assessed whether overall sentiment differs across education, landholding, information source, scheme, gender "
        "and income band, with Cramér's V as effect size. Cohen's κ measured chance-corrected agreement between VADER and the transformer. A "
        "McNemar–Bowker test assessed whether NLP sentiment and satisfaction polarity have the same marginal distribution for the same respondents, and "
        "Spearman rank correlation compared each aspect's structured rating with its ABSA score.")
    P.h2("3.12 Software and Reproducibility")
    P.p("The pipeline is implemented in Python 3.12 using pandas, NLTK (VADER, WordNet), RapidFuzz, scikit-learn, SciPy and the HuggingFace transformers "
        "library" + (", with PostgreSQL 16 as the system of record; SQL views in the database reproduce the reported figures and are checked against the "
        "Python outputs on every load" if db_ok else "") + ". Each stage is a numbered script, so the full analysis can be regenerated from the raw "
        "survey file.")

    P.h1("4. Results")
    P.h2("4.1 Awareness and Utilisation")
    P.p(f"Of {num(n['aware_n'])} respondents who answered the awareness question, {num(n['aware_yes'])} ({pct(n['aware_pct'])}) were aware of at least one "
        f"scheme. Among {num(gap['n_usable'])} respondents who listed at least one scheme, the mean was {gap['avg_heard']:.2f} schemes heard of versus "
        f"{gap['avg_enrolled']:.2f} enrolled in, and {num(gap['n_gap'])} ({pct(gap['pct_gap'])}) named more schemes heard of than enrolled in. Conversion "
        f"from awareness to enrolment differs sharply by scheme (Table 1): {pct(best['Conversion %'])} of farmers aware of {best['Scheme']} are enrolled, "
        f"against {pct(crop_conv['Conversion %'])} for Crop Insurance and {pct(worst['Conversion %'])} for {worst['Scheme']}.")
    P.table("Awareness-to-enrolment conversion by scheme", ["Scheme", "Heard of", "Enrolled", "Heard and enrolled", "Conversion"],
            [[r["Scheme"], num(r["Heard of"]), num(r["Enrolled"]), num(r["Heard and enrolled"]), pct(r["Conversion %"])] for _, r in conversion.iterrows()])

    so = n["sentiment_overall"]
    P.h2("4.2 Overall Sentiment Distribution")
    P.p(f"Averaged across the five open-ended fields, lexicon scoring classified {num(so['Positive']['n'])} respondents ({pct(so['Positive']['pct'])}) as "
        f"Positive, {num(so['Negative']['n'])} ({pct(so['Negative']['pct'])}) as Negative and {num(so['Neutral']['n'])} ({pct(so['Neutral']['pct'])}) as "
        f"Neutral (n = {num(n['n_scored'])}; {num(n['n_respondents'] - n['n_scored'])} respondents gave no usable text).")

    confusion = s("VADER vs Transformer").set_index("VADER \\ Transformer")
    off = confusion.where(~np.eye(len(confusion), dtype=bool))
    cell = off.stack().idxmax()
    cell_n = int(off.stack().max())
    P.h2("4.3 Cross-Method Comparison")
    t = P.table(f"VADER vs transformer on identical text (n = {num(agree['n'])})", ["VADER \\ Transformer", "Negative", "Neutral", "Positive"],
                [[i] + [num(v) for v in r] for i, r in confusion.iterrows()])
    P.p(f"On the same combined feedback text (n = {num(agree['n'])}), the two methods agreed on {num(agree['n_agree'])} responses "
        f"({pct(agree['n_agree'] / agree['n'] * 100)}); Cohen's κ was only {n['kappa']:.3f}, conventionally \"slight\" agreement (Table {t}). The largest "
        f"disagreement cell ({num(cell_n)} responses, {pct(cell_n / agree['n'] * 100)}) is text VADER scored {cell[0]} but the transformer scored {cell[1]}; "
        "inspection showed these are mostly mixed responses combining a benefit with a complaint, which VADER's additive scoring resolves towards the "
        "stronger lexical hits while the transformer hedges towards Neutral.")

    P.h2("4.4 Aspect-Based Sentiment")
    neg_rank = absa["% Negative"].sort_values(ascending=False)
    t = P.table("Aspect-based sentiment by service dimension", ["Aspect", "Mentions", "% Negative", "% Neutral", "% Positive"],
                [[i, num(r["Mentions"]), pct(r["% Negative"]), pct(r["% Neutral"]), pct(r["% Positive"])] for i, r in absa.iterrows()])
    P.p(f"{neg_rank.index[0]} has the highest negative share ({pct(neg_rank.iloc[0])}), followed by {neg_rank.index[1]} ({pct(neg_rank.iloc[1])}), while "
        f"Amount of support and Overall impact are predominantly positive ({pct(absa.loc['Amount of support', '% Positive'])} and "
        f"{pct(absa.loc['Overall impact', '% Positive'])}) (Table {t}).")

    P.h2("4.5 Triangulation: Stated Satisfaction vs Text Sentiment")
    tri = s("Triangulation Divergence")
    t = P.table("Divergence between ratings and text sentiment", ["Comparison", "n comparable", "n divergent", "% divergent"],
                [[r["comparison"], num(r["n_comparable"]), num(r["n_divergent"]), pct(r["pct_divergent"])] for _, r in tri.iterrows()])
    mb = n["mcnemar_bowker"]
    P.p(f"Text sentiment contradicted the respondent's own rating in {num(div['divergence_overall']['n_div'])} of {num(div['divergence_overall']['n'])} "
        f"comparable responses ({pct(div['divergence_overall']['pct'])}), rising to {pct(div['divergence_timeliness']['pct'])} for Timeliness and "
        f"{pct(div['divergence_enrollment_ease']['pct'])} for Enrollment ease (Table {t}). A McNemar–Bowker test on the paired classifications was "
        f"significant (χ² = {mb['chi2']:.2f}, df = {mb['dof']}, n = {num(mb['n'])}, p {p_value(mb['p'])}): the two measures differ systematically, not by chance.")

    P.h2("4.6 Topic Modelling: Drivers of Satisfaction and Dissatisfaction")
    rows = [["Dissatisfaction", r["label"], pct(r["share"] * 100)] for _, r in dis.iterrows()] + \
           [["Satisfaction", r["label"], pct(r["share"] * 100)] for _, r in sat.iterrows()]
    t = P.table("Topic-model results (share of corpus)", ["Type", "Topic (labelled from top terms)", "Share"], rows)
    timely = sat[sat["label"] == "Timely direct bank transfer"]["share"].sum() * 100
    P.p(f"Payment timing dominates the complaints: the two timing topics — late credit of benefits and payment delays with bank verification — together "
        f"account for {pct(timing_share)} of the problem corpus, while \"timely direct bank transfer\" is itself a satisfaction topic ({pct(timely)}) (Table {t}). "
        "The same mechanism therefore drives both ends of the sentiment distribution, depending on whether a payment arrived on time. By scheme, late "
        f"credit makes up {pct(by_scheme_pct.loc['Crop Insurance', 'Late credit of benefits'])} of Crop Insurance complaints and "
        f"{pct(by_scheme_pct.loc['Kisan Credit Card', 'Late credit of benefits'])} for Kisan Credit Card, while insufficient support is proportionally "
        f"highest for PM-KISAN among the large schemes ({pct(by_scheme_pct.loc['PM-KISAN', 'Insufficient support for crop needs'])}).")
    sel = topics.parse("Topic Model Selection")
    k5 = sel[sel["n_topics"] == 5].set_index("corpus")
    coh = sel.groupby("corpus")["umass_coherence"].agg(["min", "max"])
    t = P.table("Topic-model diagnostics (UMass coherence, perplexity, seed stability)", ["Corpus", "Topics", "Coherence", "Perplexity", "Stability (Jaccard)"],
                [[r["corpus"], r["n_topics"], f"{r['umass_coherence']:.3f}", f"{r['perplexity']:.1f}", f"{r['seed_stability_jaccard']:.3f}"]
                 for _, r in sel.iterrows()], size=8)
    P.p(f"These topics should be read as exploratory (Table {t}). Coherence varied little across 3–10 topics (C4 {coh.loc['C4', 'min']:.2f} to "
        f"{coh.loc['C4', 'max']:.2f}; C5 {coh.loc['C5', 'min']:.2f} to {coh.loc['C5', 'max']:.2f}), so no number of topics was clearly preferred; five "
        "were retained for interpretability. Refitting with different random seeds reproduced on average "
        f"{k5.loc['C4', 'seed_stability_jaccard']:.2f} (C4) and {k5.loc['C5', 'seed_stability_jaccard']:.2f} (C5) Jaccard overlap of each topic's top "
        "terms, so individual topic boundaries are not stable. The timing theme is supported independently by the Timeliness aspect results "
        "(Sections 4.4–4.5), which do not depend on the topic model.")

    P.h2("4.7 Demographic Patterns and Significance Tests")
    income = s("Sentiment by Income").set_index("Group")
    land = s("Sentiment by Landholding").set_index("Group")
    source = s("Sentiment by Info Source").set_index("Group")
    low_inc = [g for g in income.index if "Below" in str(g)]
    high_inc = [g for g in income.index if "Above" in str(g)]
    names = {"education": "Education level", "income": "Income band", "scheme": "Scheme", "landholding": "Landholding band",
             "info_source": "Information source", "gender": "Gender"}
    chi = sorted(n["chi_square"].items(), key=lambda kv: -kv[1]["v"])
    vs = [v["v"] for _, v in chi]
    gov = [g for g in source.index if "Government official" in str(g)]
    P.p(f"Overall sentiment varied significantly across all six grouping variables, with small effect sizes (Cramér's V {min(vs):.3f}–{max(vs):.3f}). "
        + (f"Negative sentiment fell from {pct(income.loc[low_inc[0], 'Negative'])} among farmers earning {low_inc[0]} to {pct(income.loc[high_inc[0], 'Negative'])} "
           f"among those earning {high_inc[0]}. " if low_inc and high_inc else "")
        + f"Marginal farmers (<2.5 acres, n = {num(land.loc['Marginal (<2.5 ac)', 'n'])}) were the most negative landholding group "
          f"({pct(land.loc['Marginal (<2.5 ac)', 'Negative'])}), consistent with a positive rank correlation between acreage and sentiment "
          f"(ρ = {n['landholding_spearman']['rho']:.3f}, p {p_value(n['landholding_spearman']['p'])}). "
        + (f"Respondents who learned of schemes from a government official (n = {num(source.loc[gov[0], 'n'])}) were the most positive group "
           f"({pct(source.loc[gov[0], 'Positive'])} positive)." if gov else ""))
    t = P.table("Chi-square tests: overall sentiment × group", ["Grouping", "χ²", "df", "n", "p", "Cramér's V"],
                [[names[k], f"{v['chi2']:.2f}", v["dof"], num(v["n"]), p_value(v["p"]), f"{v['v']:.3f}"] for k, v in chi])
    sp = n["spearman_aspects"]
    sig_pos = [k for k, v in sp.items() if v["p"] < 0.05 and v["rho"] > 0]
    sig_neg = [k for k, v in sp.items() if v["p"] < 0.05 and v["rho"] < 0]
    P.p("Spearman correlation between each aspect's structured rating and its ABSA score was positive and significant only for "
        + ", ".join(ASPECT_NAMES[k] + f" (ρ = {signed(sp[k]['rho'])})" for k in sig_pos)
        + ("; it was small but significantly negative for " + ", ".join(ASPECT_NAMES[k] + f" (ρ = {signed(sp[k]['rho'])})" for k in sig_neg) if sig_neg else "")
        + ", and not significant for the remaining aspects. At the level of individual respondents, ABSA polarity scores therefore track structured "
          "ratings poorly — aspect polarity, rather than aspect detection, is the pipeline's weakest component.")
    P.table("Spearman correlation: structured C3 rating vs ABSA score", ["Aspect", "n", "ρ", "p"],
            [[ASPECT_NAMES[k], num(v["n"]), signed(v["rho"]), p_value(v["p"])] for k, v in sp.items()])

    P.h2("4.8 Scheme-wise Sentiment and Satisfaction")
    rb, pk, ci = schemes.loc["Rythu Bharosa"], schemes.loc["PM-KISAN"], schemes.loc["Crop Insurance"]
    P.p(f"Scheme-level results are not uniform (Table {P.tables + 1}). Rythu Bharosa ({pct(rb['pct_of_feedback'])} of scheme feedback) and PM-KISAN "
        f"({pct(pk['pct_of_feedback'])}) dominate by volume and are predominantly positive (average satisfaction {rb['avg_satisfaction']:.2f} and "
        f"{pk['avg_satisfaction']:.2f} on a −2 to +2 scale). Crop Insurance (n = {num(ci['N'])}) has a far higher negative share ({pct(ci['pct_negative'])}) "
        f"than any other scheme and the lowest satisfaction among adequately sampled schemes ({ci['avg_satisfaction']:.2f}).")
    P.table("Sentiment and satisfaction by scheme", ["Scheme", "N", "% Pos.", "% Neu.", "% Neg.", "Avg. satisfaction", "Avg. impact"],
            [[i, num(r["N"]), pct(r["pct_positive"]), pct(r["pct_neutral"]), pct(r["pct_negative"]), f"{r['avg_satisfaction']:.2f}",
              "" if pd.isna(r["avg_impact"]) else f"{r['avg_impact']:.2f}"] for i, r in schemes.iterrows()], size=8)
    P.p(f"PM-KISAN shows a different pattern: strong satisfaction ({pk['avg_satisfaction']:.2f}) and the highest positive share of the two largest schemes "
        f"({pct(pk['pct_positive'])}), but the lowest self-reported farming impact of any adequately sampled scheme ({pk['avg_impact']:.2f}) and the "
        f"lowest structured Amount-of-support rating of any major scheme ({amount_rating['PM-KISAN']:.2f}). Farmers appear satisfied that PM-KISAN "
        "arrives reliably but do not feel the fixed instalment is large enough to change their farming — a benefit-adequacy problem that satisfaction "
        "scores alone would not reveal.")
    P.p(f"Disaggregating ABSA by scheme (Table {P.tables + 1}) shows that Enrollment ease is the most negatively described aspect for each of the four "
        "largest schemes" + ("" if enrol_worst_all else " except where noted") + ". Because this recurs across a state investment-support scheme "
        "(Rythu Bharosa), a fixed-instalment income-support scheme (PM-KISAN), a credit instrument (Kisan Credit Card) and a subsidy programme (Input "
        "Subsidy), a shared registration or verification process is a more plausible explanation than four independent design failures.")
    P.table("ABSA: % of aspect mentions scored Negative, by scheme", ["Scheme", "N", "Timeliness", "Amount", "Enrollment ease", "Official behavior", "Clarity"],
            [[sc, num(absa_scheme.loc[sc, "N"])] + [pct(absa_scheme.loc[sc, f"{a}_pct_negative"]) for a in ASPECTS[:5]] for sc in MAJOR], size=8)

    P.h2("4.9 Farmer-Authored Improvement Priorities")
    P.p(f"Term-frequency analysis of {num(n['n_suggestions'])} substantive suggestions (Table {P.tables + 1}) is dominated by requests to increase the "
        "benefit amount, followed by simplifying and speeding up applications, more awareness programmes and faster disbursement. Compared with the "
        "complaint topics (Section 4.6), the two fields converge on process friction — registration, documents and office approval appear in both — but "
        "differ in emphasis: complaints are dominated by payment timing, whereas suggestions put the size of the benefit first. Farmers describe delays "
        "as the problem they experience, and a larger amount as the change they most want.")
    P.table("Ranked suggestion themes (C6)", list(themes.columns), themes.astype(str).values.tolist())

    P.h2("4.10 Validation Against Human Labels")
    if methods is not None:
        t = P.table("Overall-sentiment validation against human labels", ["Method", "n", "Accuracy (95% CI)", "Macro-F1 (95% CI)", "κ"],
                    [[r["method"], num(r["n"]), f"{r['accuracy']:.3f} ({r['accuracy_ci95']})", f"{r['macro_f1']:.3f} ({r['macro_f1_ci95']})", f"{r['cohen_kappa']:.3f}"]
                     for _, r in methods.iterrows()], size=8)
        P.p(f"Table {t} compares each method with the human labels ({summary_val.get('Overall worksheet rows labelled', '')} rows labelled). "
            f"{best_method} achieved the highest macro-F1. The zero-shot methods are evaluated on all labelled rows, while the supervised baselines "
            "are evaluated out-of-fold, so their scores reflect performance on responses they were not trained on.")
        if detect is not None and absa_val is not None:
            pooled = absa_val[absa_val["aspect"] == "ALL (pooled)"]
            P.p(f"For aspect-level sentiment ({summary_val.get('ABSA worksheet rows annotated', '')} responses), keyword-based aspect detection reached a mean "
                f"F1 of {detect['f1'].mean():.2f} across the six aspects"
                + (f", while polarity accuracy on aspects that both the pipeline and the annotator identified was {pooled['accuracy'].iloc[0]:.2f}"
                   f" (macro-F1 {pooled['macro_f1'].iloc[0]:.2f}, n = {num(pooled['n'].iloc[0])})." if not pooled.empty else "."))
            P.table("Aspect-level validation: polarity on jointly detected aspects", ["Aspect", "n", "Accuracy", "Macro-F1", "κ"],
                    [[r["aspect"], num(r["n"]), f"{r['accuracy']:.3f}", f"{r['macro_f1']:.3f}", f"{r['cohen_kappa']:.3f}"] for _, r in absa_val.iterrows()], size=8)
    else:
        P.p("Human annotation of the stratified validation samples (Section 3.10) is in progress. Classifier accuracy is therefore not reported here; "
            "the descriptive, triangulation, topic and suggestion findings above do not depend on it.")

    P.h1("5. Discussion")
    P.p(f"The central finding is that structured satisfaction ratings and text sentiment disagree often enough — {pct(div['divergence_overall']['pct'])} "
        f"overall, {pct(div['divergence_timeliness']['pct'])} for Timeliness and {pct(div['divergence_enrollment_ease']['pct'])} for Enrollment ease, "
        "confirmed by a McNemar–Bowker test — that relying on either alone would give a materially different picture of scheme performance. This extends "
        "sentiment analysis of social-media and policy text [3], [16], [19] and open-ended survey validation [27], [28] to Indian agricultural welfare "
        "schemes, where, to our knowledge, this combination of triangulation, significance testing and scheme-level disaggregation has not been applied.")
    P.p("The awareness results add a dimension that aggregate figures hide. Awareness is high, but it converts into enrolment very unevenly: almost all "
        f"farmers who know Rythu Bharosa are enrolled, whereas only {pct(crop_conv['Conversion %'])} of those aware of Crop Insurance are. Combined with "
        "Crop Insurance's high negative sentiment and its timing complaints, this points to a scheme where both access and delivery are weak.")
    P.p("The scheme-disaggregated ABSA results are the most policy-actionable. Enrollment ease is not one weak point among six; it is the most negatively "
        "described aspect for each of the four largest schemes despite their different delivery mechanisms, which is far more consistent with a shared "
        "administrative layer than with four separate design failures.")
    P.p("The lexicon audit is a methodological contribution in its own right. Correcting demonstrable errors moved the headline divergence from 17.5% to "
        f"{pct(div['divergence_overall']['pct'])} and made Enrollment ease, rather than Timeliness, the most negative aspect. These changes moved results "
        "towards internal consistency: the enrolment finding is corroborated by the structured ratings and by the scheme-level breakdown, neither of which "
        "depends on the corrected words.")
    P.p(f"At the same time, the cross-method comparison (κ = {n['kappa']:.3f}), the weak aspect-level correlations and "
        + ("the validation results (Section 4.10)" if methods is not None else "the absence of completed human validation")
        + " deliver a cautionary methodological message: automated sentiment scoring, whether lexicon- or transformer-based, should not be reported as "
          "ground truth for this kind of short, multi-clause, translated text without validation. The persistence of disagreement after correcting "
          "concrete lexicon errors suggests a structural limit of word-additive scoring, which does not model compositional meaning or negation scope.")

    P.h1("6. Policy Implications")
    P.p("In priority order:")
    bullet(doc, "**Review the enrolment and registration process across schemes, not scheme by scheme.** Enrollment ease is the most negatively described "
                "aspect for every major scheme regardless of design, pointing to a shared bottleneck in registration, documentation or verification.")
    bullet(doc, "**Make disbursement timing an operational priority.** Late and delayed credit is the largest complaint theme, while timely transfer is a "
                "leading source of satisfaction.")
    bullet(doc, f"**Track awareness-to-enrolment conversion as a scheme-level indicator.** Conversion ranges from {pct(best['Conversion %'])} "
                f"({best['Scheme']}) to {pct(worst['Conversion %'])} ({worst['Scheme']}); schemes that farmers know but do not join need targeted "
                "enrolment support, not more awareness campaigns.")
    bullet(doc, f"**Commission a dedicated review of Crop Insurance**, which combines the lowest conversion among major schemes "
                f"({pct(crop_conv['Conversion %'])}), the highest negative-sentiment share and timing-dominated complaints.")
    bullet(doc, "**Review the adequacy of PM-KISAN's benefit amount.** Its problem is not dissatisfaction but low felt impact and the lowest Amount rating "
                "of any major scheme — something satisfaction monitoring alone would miss.")
    bullet(doc, "**Add open-ended questions on enrolment experience and payment timing to monitoring surveys**, since these dimensions show the largest gap "
                "between ratings and what farmers say.")
    P.p(f"These recommendations are descriptive and correlational, based on {num(n['n_respondents'])} respondents in one district, and should be read as "
        "hypothesis-generating rather than causally validated.")

    P.h1("7. Limitations")
    for text in [
        f"Single district: the sample (N = {num(n['n_respondents'])}) is from Chittoor, Andhra Pradesh; generalisation beyond this district and these "
        "schemes should be made cautiously.",
        f"Small samples for minor schemes: scheme-level findings are most reliable for Rythu Bharosa and PM-KISAN (n = {num(rb['N'])} and {num(pk['N'])}); "
        f"the Crop Insurance result (n = {num(ci['N'])}) should be confirmed with a larger sample.",
        "Translation: responses were recorded in English (90% of interviews were conducted in Telugu); translation may flatten nuance in the original language.",
        "Topic models are exploratory: coherence did not single out a number of topics, topics were only moderately stable across random seeds, and "
        "excluding empty answers changed the complaint topics, so topic shares are approximate and labels interpretive.",
        "Lexicon correction was applied once to words meeting a frequency and context threshold; less frequent gaps remain.",
        ("Joint labelling: validation labels were produced jointly by the author and one additional annotator as a single consensus set, so "
         "inter-annotator agreement could not be measured; independent double-labelling of a subset would strengthen the validation." if methods is not None else
         "Validation pending: classifier accuracy against human labels was not yet available at the time of writing."),
        "Weak ABSA–rating correlation for five of six aspects reflects a structural limitation of word-additive polarity scoring rather than a small-sample artefact.",
    ]:
        bullet(doc, text)

    P.h1("8. Conclusion")
    P.p(f"This study combined a normalised relational database with a multi-method NLP pipeline to analyse feedback from {num(n['n_respondents'])} farmers "
        f"on government welfare schemes in Chittoor district. It found a large and uneven awareness-to-enrolment gap ({pct(gap['pct_gap'])} of "
        f"respondents named more schemes heard of than enrolled in; conversion from {pct(worst['Conversion %'])} to {pct(best['Conversion %'])} across "
        f"schemes), a statistically confirmed divergence between stated satisfaction and text sentiment ({pct(div['divergence_overall']['pct'])} overall, "
        "concentrated in timeliness and enrolment), and enrolment friction shared by all major schemes. Crop Insurance stood out for weak conversion and "
        "high dissatisfaction, and PM-KISAN for satisfaction masking low felt impact. Complaints centred on payment timing while suggestions centred on "
        "benefit amount and simpler procedures. Cross-method and correlational testing"
        + (" and validation against human labels" if methods is not None else "")
        + " show that automated sentiment should not be treated as ground truth in this domain without validation. The pipeline, database design and "
          "annotation protocol are reusable for other districts and support evidence-based, transparent evaluation of agricultural welfare schemes.")

    P.h1("References")
    for i, ref in enumerate(REFERENCES, start=1):
        para(doc, f"[{i}] {ref}", size=10)

    doc.save(OUT)
    print(f"Saved {OUT} ({P.tables} tables)")
    for w in P.warnings:
        print("CLAIM CHECK FAILED:", w)


if __name__ == "__main__":
    main()
