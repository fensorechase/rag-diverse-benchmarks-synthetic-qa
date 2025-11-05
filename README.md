# rag-diverse-benchmarks-synthetic-qa

This repository contains our implementation for the short paper manuscript "Designing Diverse RAG Benchmarks: A Hierarchical Framework for Synthetic Question Generation"

**For expanded tables and extended metrics -- please see [Extended Results](./extended_results.md).**

## Installation

```bash
# Clone repository
git clone (from https://anonymous.4open.science/r/rag-diverse-benchmarks-synthetic-qa-3E08)
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

Also, ensure you populate your .env file with your Huggingface and ai71 tokens.

## Usage

1. You may choose to generate synthetic QAs using DataMorgana, if you have an ai71 account.

```bash
cd synthetic_qa_generate
```

... and run steps in Driver_Synthetic_QA_RAG_Benchmark.ipynb

2. Then, on a GPU-capable system, generate answers using the baseline RAG system (BM25, Flacon-3-10B-Instruct)

- If desired, you may adjust your RAG, language model system settings in rag_system/generate/config.py, and the shell script gen_temp.sh.

The shell script runs answer generation in parallel for synthetic QAs in RQ1, RQ2, RQ3:

```bash
cd rag_system/generate
sbatch gen_temp.sh
```

3. To analyze results from this answer generation, inside rag_system/generate, run:

```bash
sbatch analyze_temp.sh
```

## Project Structure

```
rag_system/
├── generate/                       # Answer generation
├── config.py                       # Configuration settings
├── main.py                         # Main execution script
├── retriever_utils.py              # Retrieval components
├── utils.py                        # Utility functions
├── complete_analysis.py            # Evaluation script
├── compute_rag_metrics.py          # Evaluation script
├── analyze_rq3_interactions.py     # Evaluation script
├── complete_analysis_results.json  # Results of full evaluation 
├── results/                        # Results directory
├── rq1_results_total.jsonl
├── rq2_results_total.jsonl
└── rq3_results_total.jsonl

synthetic_qa_generate/
├── Driver_Synthetic_QA_RAG_Benchmark.ipynb # Driver notebook - all synthetic data generation via DataMorgana
├── example_DataMorgana_pipeline/ # 10 eg. prompts used to generate 100 synthetic QAs with DataMorgana. 
├── QAs_RQ1/ # Contains rq1_total_all_questions.jsonl
├── QAs_RQ2/ # Contains rq2_total_all_questions.jsonl
├── QAs_RQ3/ # Contains rq3_total_all_questions.jsonl
├── categorization_configs.py
├── EXTRA_categorization_configs.py
└── README_qagen.md # Additional details for the synthetic QA generation process.
```
