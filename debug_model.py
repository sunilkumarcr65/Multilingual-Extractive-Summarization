from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small"
)

for name, module in model.named_modules():

    if "SelfAttention" in name:
        print(name)