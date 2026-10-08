from __future__ import annotations

import re
from typing import Dict, List, Tuple


CLAUSE_PATTERNS: Dict[str, List[str]] = {
    "Services": [
        r"\bservices?\b",
        r"\bstatement of work\b",
        r"\bscope of work\b",
    ],
    "Payments": [
        r"\bfees?\b",
        r"\bpayment\b",
        r"\breimburse\b",
        r"\binvoice\b",
    ],
    "Relationship": [
        r"\bindependent contractor\b",
        r"\bemployee\b",
        r"\bagent\b",
    ],
    "IP Ownership": [
        r"\bintellectual property\b",
        r"\bwork product\b",
        r"\bassign(?:ment)?\b",
        r"\bmoral rights?\b",
    ],
    "Confidentiality": [
        r"\bconfidential\b",
        r"\bnon-?disclosure\b",
        r"\bconfidential information\b",
    ],
    "Data Protection": [
        r"\bgdpr\b",
        r"\bccpa\b",
        r"\bpersonal data\b",
        r"\bdata breach\b",
    ],
    "Termination": [
        r"\btermination\b",
        r"\bterm\b",
        r"\bnotice period\b",
        r"\bbreach\b",
    ],
    "Liability": [
        r"\bindemnif(?:y|ication)\b",
        r"\blimitation of liability\b",
        r"\bliab(?:ility|le)\b",
    ],
    "Governing Law": [
        r"\bgoverning law\b",
        r"\barbitration\b",
        r"\bjams\b",
    ],
}


RISK_RULES: List[Tuple[str, int, str]] = [
    (
        r"\bsole discretion\b",
        12,
        "Broad unilateral discretion language appears.",
    ),
    (
        r"\bwithout assigning any reason\b",
        14,
        "Termination without-cause language detected.",
    ),
    (
        r"\battorney-?in-?fact\b",
        10,
        "Power-of-attorney clause detected for IP actions.",
    ),
    (
        r"\bno further fees? or compensation\b",
        10,
        "Compensation-limiting language detected.",
    ),
    (
        r"\bmonitor(?:ing)?\b.{0,80}\bcommunications?\b",
        8,
        "Monitoring of communications is stated.",
    ),
]


def _split_sentences(text: str) -> List[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text or "")
        if sentence.strip()
    ]


def build_contract_insights(text: str) -> Dict:
    text = (text or "").strip()
    if not text:
        return {
            "clause_map": [],
            "risk_score": 0,
            "risk_level": "Low",
            "risk_flags": [],
        }

    sentences = _split_sentences(text)

    clause_map = []
    for clause_name, patterns in CLAUSE_PATTERNS.items():
        evidence = []
        for sentence in sentences:
            lower_sentence = sentence.lower()
            if any(re.search(p, lower_sentence) for p in patterns):
                evidence.append(sentence)
            if len(evidence) >= 2:
                break
        clause_map.append(
            {
                "clause": clause_name,
                "found": bool(evidence),
                "evidence": evidence,
            }
        )

    risk_score = 0
    risk_flags = []
    lower_text = text.lower()
    for pattern, weight, message in RISK_RULES:
        if re.search(pattern, lower_text):
            risk_score += weight
            risk_flags.append(message)

    risk_score = max(0, min(100, risk_score))
    if risk_score >= 50:
        risk_level = "High"
    elif risk_score >= 25:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "clause_map": clause_map,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_flags": risk_flags,
    }

