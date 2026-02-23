# Synthetic QA Generation

Two ways to generate synthetic QA pairs. The pre-generated datasets are already in `QAs_RQ1/`, `QAs_RQ2/`, `QAs_RQ3/` — you only need this if you want to regenerate or adapt the benchmark.

---

## Option A: DataMorgana API (exact replication)

Requires an [ai71](https://platform.ai71.ai) account and API key. DataMorgana offers 10,000 free credits.

1. Add your ai71 API key to `.env`.
2. Set your desired categorizations in `categorization_configs.py`.
3. Run `Driver_Synthetic_QA_RAG_Benchmark.ipynb` — cells are organized by RQ (RQ1 granularity, RQ2 complementarity, RQ3 interactions).

The notebook calls the DataMorgana `bulk_generation` endpoint, which randomly samples documents from the FineWeb-10BT corpus server-side. To target specific document IDs, pass them in the `document_ids` field of the API request.

---

## Option B: Local pipeline (no API, any corpus)

See **`local_datamorgana_pipeline/README_local_datamorgana.md`** for full instructions.

Replicates the DataMorgana generation logic locally using any HuggingFace model and your own corpus of `.txt` or `.jsonl` files. Output format is identical to Option A and compatible with all downstream RAG evaluation scripts.

---

## Categorization configs

| File | Contents |
|---|---|
| `categorization_configs.py` | All question and user categorizations for RQ1–RQ3 (coarse/medium/fine granularity, complementarity sets, interaction factorial) |
| `EXTRA_categorization_configs.py` | Additional fine-grained categories added in the second generation round |

To define new categories, copy any existing block and change the `name`, `description`, and `probability` fields. Probabilities within a categorization must sum to 1.0.

---

## Output format

Each generated QA pair (JSONL, one object per line):

```json
{
  "question": "...",
  "answer": "...",
  "context": ["..."],
  "question_categories": [{"categorization_name": "...", "category_name": "..."}],
  "user_categories":     [{"categorization_name": "...", "category_name": "..."}],
  "document_ids": ["<urn:uuid:...>"]
}
```