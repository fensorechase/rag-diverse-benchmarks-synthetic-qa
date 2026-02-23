# Local DataMorgana-Style Synthetic QA Generation

Replicates the [DataMorgana](https://arxiv.org/abs/2501.12789) synthetic QA pipeline locally — no ai71 account, no API key, no pre-indexed corpus required. Uses any HuggingFace instruction-tuned model and your own documents.

Output format is identical to the DataMorgana API and compatible with all RAG evaluation scripts in `rag_system/`.

---

## Files

```bash
local_datamorgana_pipeline/
├── datamorgana_local.py           # Core pipeline (sampling, prompting, filtering, generation)
├── Driver_Local_DataMorgana.ipynb # Interactive notebook with dry-run cells
├── requirements.txt
├── configs/
│   ├── default_config.json        # 4 question + 1 user categorization (matches the paper)
│   ├── granularity_config.json    # Coarse/medium/fine hierarchy for RQ1
│   └── healthcare_config.json     # Domain-specific user roles example
└── sample_corpus/
    ├── doc_001.txt                # 3 example documents
    ├── doc_002.txt
    ├── doc_003.txt
    └── synthetic_qa_output.jsonl  # Example output (3 QA pairs)
```

---

## Quick start

```bash
pip install -r requirements.txt

python datamorgana_local.py \
    --corpus_dir  ./sample_corpus \
    --config      configs/default_config.json \
    --output      ./my_qa.jsonl \
    --n_questions 200 \
    --model       meta-llama/Meta-Llama-3.1-8B-Instruct
```

Or open `Driver_Local_DataMorgana.ipynb` for a step-by-step interactive version with prompt inspection before running the full batch.

---

## Corpus

Put your documents in any directory as `.txt` files (one doc per file) or `.jsonl` files (one object per line with a `"text"` field).

To replicate the FineWeb-10BT corpus used in the paper, fetch a small sample:

```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
for i, row in enumerate(ds):
    with open(f"sample_corpus/doc_{i+1:03d}.txt", "w") as f:
        f.write(row["text"])
    if i == 9: break
```

10 documents (~50 KB) is sufficient for testing; scale up as needed.

---

## Recommended models

| Model | VRAM | Notes |
|---|---|---|
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 16 GB | Best general-purpose choice |
| `mistralai/Mistral-7B-Instruct-v0.3` | 14 GB | Fast and reliable |
| `tiiuae/Falcon3-10B-Instruct` | 20 GB | Model used in the paper |
| `Qwen/Qwen2.5-7B-Instruct` | 14 GB | Strong instruction-following |

Add `--load_in_4bit` (requires `bitsandbytes`) to halve VRAM requirements.

---

## Key arguments

| Argument | Default | Description |
|---|---|---|
| `--corpus_dir` | `./sample_corpus` | Directory of `.txt` or `.jsonl` documents |
| `--config` | `configs/default_config.json` | Categorization config file |
| `--output` | `./synthetic_qa.jsonl` | Output path |
| `--n_questions` | 200 | QA pairs to generate |
| `--k_candidates` | 3 | Candidates per document before filtering |
| `--model` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | HuggingFace model name or local path |
| `--load_in_4bit` | off | 4-bit quantisation |
| `--seed` | 42 | Random seed |

---

## Defining categorizations

Edit any config JSON or define inline in the notebook. Rules from the DataMorgana paper:

- Categories within a categorization must be mutually exclusive.
- Probabilities must sum to 1.0.
- The `description` field is injected verbatim into the LLM prompt — write it carefully.

`granularity_config.json` contains all coarse/medium/fine category blocks from the extended results as ready-to-use drop-ins.

---

## Differences from the DataMorgana API

| | DataMorgana API | This replication |
|---|---|---|
| Document source | FineWeb-10BT (server-side) | Your corpus directory |
| LLM | Proprietary Falcon-based | Any HuggingFace model |
| Faithfulness filter | NLI-based (server-side) | Keyword-overlap heuristic |
| Scale | Async, thousands of questions | Sequential; parallelize via shell |

The prompt template is taken verbatim from §3.2 of Filice et al. (2025).
