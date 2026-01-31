import re
from typing import Dict, Any
import pandas as pd
import regex as re


# --- Regex patterns ---
#Match the format of QEvasion set
# Matches leading speaker labels like:
# "THE PRESIDENT.", "THE PRESIDENT:", "PRESIDENT BIDEN:", "PRESIDENT TRUMP —", "PRESIDENT **"
SPEAKER_PREFIX_RE = re.compile(
    r"""
    ^\s*
    (?:
        the\s+president
        |president
        |the\s+vice\s+president
        |vice\s+president
    )
    (?:\s+(?:\p{L}[\p{L}\p{M}\-']*|\p{L}\.|\*+|\([^\)]*\))){0,8}
    \s*
    (?:[.:\-\u2013\u2014]+|\*+)? 
    \s*
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Capitalized name tokens only (prevents removing sentences)
LEFTOVER_NAME_PREFIX_RE = re.compile(
    rf"""
    ^\s*
    (?:[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ'\-]+\s+){{0,2}}
    [A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ'\-]+
    \s*[\.:]\s+
    """,
    re.VERBOSE,
)


def clean_answer(text: str) -> str:
    if text is None:
        return text

    t = str(text).strip()
    # 1) Remove explicit speaker tags
    t = SPEAKER_PREFIX_RE.sub("", t, count=1)
    # 2) Remove leftover NAME fragments only (not sentences)
    t = LEFTOVER_NAME_PREFIX_RE.sub("", t, count=1)
    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def preprocess_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    ex["interview_answer"] = clean_answer(ex.get("interview_answer"))
    return ex



df = pd.read_csv("clarity_task_evaluation_dataset.csv")

df["interview_answer"] = df["interview_answer"].apply(clean_answer)


df.to_csv("clarity_task_evaluation_dataset_cleaned.csv", index=False)
print("Saved:", "clarity_task_evaluation_dataset_cleaned.csv")
