from __future__ import annotations

from functools import lru_cache

import nltk


# ---------------------------------------------------------
# Resource Map
# ---------------------------------------------------------

NLTK_RESOURCES = {

    "punkt":
        "tokenizers/punkt",

    "punkt_tab":
        "tokenizers/punkt_tab",

    "wordnet":
        "corpora/wordnet",
}


# ---------------------------------------------------------
# Download
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def ensure_nltk_resources() -> None:
    """
    Ensure required NLTK resources exist.
    """

    for package, resource_path in (
        NLTK_RESOURCES.items()
    ):

        try:

            nltk.data.find(
                resource_path
            )

        except LookupError:

            print(
                f"[NLTK] "
                f"Downloading "
                f"{package}..."
            )

            try:

                nltk.download(
                    package,
                    quiet=True,
                    raise_on_error=False,
                )

            except Exception as error:

                print(
                    f"[NLTK] "
                    f"Failed to "
                    f"download "
                    f"{package}: "
                    f"{error}"
                )

    print(
        "[NLTK] Resources ready."
    )