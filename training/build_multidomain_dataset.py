from datasets import (
    concatenate_datasets,
    load_dataset,
)


# =========================================================
# HELPERS
# =========================================================

def keep_columns(example):

    return {
        "document": example["document"],
        "summary": example["summary"],
        "domain": example["domain"],
    }


# =========================================================
# XL-SUM (NEWS)
# =========================================================

print("\n[LOADING] XL-Sum")

xlsum = load_dataset(
    "csebuetnlp/xlsum",
    "english",
    trust_remote_code=True,
)["train"].select(range(2000))


def map_xlsum(example):

    return {
        "document": example["text"],
        "summary": example["summary"],
        "domain": "news",
    }


xlsum = xlsum.map(map_xlsum)

xlsum = xlsum.map(keep_columns)

xlsum = xlsum.remove_columns([
    col for col in xlsum.column_names
    if col not in [
        "document",
        "summary",
        "domain",
    ]
])

print("[DONE] XL-Sum")


# =========================================================
# BILLSUM (LEGAL)
# =========================================================

print("\n[LOADING] BillSum")

billsum = load_dataset(
    "billsum",
)["train"].select(range(1000))


def map_billsum(example):

    return {
        "document": example["text"],
        "summary": example["summary"],
        "domain": "legal",
    }


billsum = billsum.map(map_billsum)

billsum = billsum.map(keep_columns)

billsum = billsum.remove_columns([
    col for col in billsum.column_names
    if col not in [
        "document",
        "summary",
        "domain",
    ]
])

print("[DONE] BillSum")


# =========================================================
# MERGE
# =========================================================

print("\n[MERGING DATASETS]")

merged = concatenate_datasets([
    xlsum,
    billsum,
])

merged = merged.shuffle(seed=42)

print(merged)

print("\n[SAVING DATASET]")

merged.save_to_disk(
    "./training/data/merged/multidomain_dataset"
)

print("\n[SUCCESS] Multi-domain dataset created")