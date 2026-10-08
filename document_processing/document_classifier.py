from __future__ import annotations

import re


RESUME_KEYWORDS = {
    "education",
    "experience",
    "skills",
    "projects",
    "cgpa",
    "leetcode",
    "codeforces",
    "github",
}

LEGAL_KEYWORDS = {
    "agreement",
    "contractor agreement",
    "effective date",
    "statement of work",
    "governing law",
    "arbitration",
    "limitation of liability",
    "confidential information",
    "data protection",
    "term and termination",
}

RESEARCH_KEYWORDS = {
    "abstract",
    "methodology",
    "references",
    "results",
    "conclusion",
}

NEWS_KEYWORDS = {
    "reported",
    "according",
    "breaking",
    "announced",
}


def classify_document(
    text: str,
) -> str:
    """
    Lightweight document classifier.
    """

    text_lower = (
        text.lower()
    )

    # -----------------------------------------------------
    # Legal/contract docs
    # -----------------------------------------------------

    legal_matches = sum(
        keyword in text_lower
        for keyword in LEGAL_KEYWORDS
    )

    if legal_matches >= 3:
        return "general"

    # -----------------------------------------------------
    # Resume
    # -----------------------------------------------------

    resume_matches = sum(
        keyword in text_lower
        for keyword
        in RESUME_KEYWORDS
    )

    has_resume_headers = bool(
        re.search(
            r"(education|work experience|projects|skills)",
            text_lower,
        )
    )

    has_profile_signals = bool(
        re.search(
            r"(linkedin|github|leetcode|codeforces|codechef|cgpa|gpa)",
            text_lower,
        )
    )

    if (
        resume_matches >= 3
        and has_resume_headers
        and has_profile_signals
    ):
        return "resume"

    # -----------------------------------------------------
    # Research paper
    # -----------------------------------------------------

    research_matches = sum(
        keyword in text_lower
        for keyword
        in RESEARCH_KEYWORDS
    )

    if research_matches >= 3:

        return "research_paper"

    # -----------------------------------------------------
    # News
    # -----------------------------------------------------

    news_matches = sum(
        keyword in text_lower
        for keyword
        in NEWS_KEYWORDS
    )

    if news_matches >= 2:

        return "news"

    return "general"
