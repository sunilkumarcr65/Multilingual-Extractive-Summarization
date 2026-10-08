from datasets import load_dataset


def safe_load(name, *args, **kwargs):

    try:

        dataset = load_dataset(
            name,
            *args,
            **kwargs,
        )

        print(f"[SUCCESS] {name} loaded")

        return dataset

    except Exception as error:

        print(
            f"[FAILED] {name}: {error}"
        )

        return None


# ---------------------------------------------------------
# XL-Sum
# ---------------------------------------------------------

xlsum = safe_load(
    "csebuetnlp/xlsum",
    "english",
    trust_remote_code=True,
)


# ---------------------------------------------------------
# BillSum
# ---------------------------------------------------------

billsum = safe_load(
    "billsum",
)


# ---------------------------------------------------------
# SAMSum
# ---------------------------------------------------------

samsum = safe_load(
    "samsum",
)


# ---------------------------------------------------------
# arXiv
# ---------------------------------------------------------

arxiv = safe_load(
    "ccdv/arxiv-summarization",
)