# GenSumm — Multilingual Extractive & Abstractive Summarizer

**Phase 2 — Research-Grade NLP Pipeline**

A multilingual document summarization system supporting PDF, DOCX, and image inputs with a Streamlit web interface.

---

## Phase 2 Improvements

| Improvement | Details |
|---|---|
| **BART-large-CNN** | Replaced Flan-T5-base with BART-large-CNN (trained for summarization) → +5–15 ROUGE points |
| **LaBSE Embeddings** | Auto-switches to LaBSE for non-English languages (Hindi, Tamil, Bengali, etc.) |
| **TextRank Baseline** | Added as a named method for research comparison |
| **Research Dashboard** | Interactive Plotly charts from XL-Sum evaluation CSVs |
| **Live Evaluation** | ROUGE-1/2/L, BLEU, METEOR, BERTScore computed on the fly |
| **Keyword Extraction** | KeyBERT keyphrases displayed as tags |
| **PDF Download** | Download generated summary as PDF |
| **Summary History** | Last 5 summaries tracked in sidebar |
| **Code Cleanup** | Removed ~800 lines of dead commented-out code |

---

## Architecture

```
app.py                          ← Streamlit UI (2 tabs)
main.py                         ← Document summarization pipeline
│
├── extractive/
│   ├── ex_summarizer.py        ← Clustering+MMR & TextRank baseline
│   ├── clustering.py           ← KMeans / Spectral / Agglomerative / OPTICS
│   ├── selector.py             ← MMR-based sentence selection
│   └── scoring.py              ← Multi-component sentence scoring
│
├── abstractive/
│   └── abs_summarizer.py       ← BART-large-CNN (default)
│
└── gensumm/
    ├── gensumm_extractive.py   ← Text extraction, embedding (MiniLM/LaBSE), splitting
    ├── translator.py           ← MBART-50 multilingual translation
    ├── sentiment_module.py     ← XLM-RoBERTa sentiment
    ├── qa_module.py            ← RAG-based Q&A
    ├── abstractive.py          ← HF pipeline wrapper
    ├── dataset_eval.py         ← Batch evaluation on XL-Sum
    └── evaluation_summarization.py ← Per-doc evaluation metrics
```

---

## Supported Languages

English, Hindi, Spanish, French, German, Portuguese, Bengali, Marathi, Tamil, Telugu

---

## Installation

```bash
pip install -r requirements.txt
python -m nltk.downloader wordnet punkt averaged_perceptron_tagger
```

## Run

```bash
streamlit run app.py
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| ROUGE-1/2/L | Unigram/bigram/longest common subsequence overlap |
| BLEU | N-gram precision (machine translation metric) |
| METEOR | Recall-oriented, handles synonyms |
| BERTScore | Semantic similarity using BERT embeddings |

---

## Summarization Modes

| Mode | Description |
|---|---|
| **Extractive** | Selects key sentences using clustering + MMR |
| **Hybrid** | Extractive pre-selection → BART refinement |
| **Abstractive** | Direct BART generation |
| **TextRank** | Classic PageRank baseline (in evaluation) |
