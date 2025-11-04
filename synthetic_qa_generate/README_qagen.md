# Synthetic Question-Answer Pair Generation Steps

Resources:
Synthetic QA generation tool: https://platform.ai71.ai/documentation. See "Synthetic Conversation Generation Using cURL", "completion," and "Synthetic Conversations".
Document corpus: we have a privately-run database with all documents in the FineWeb-10BT corpus, and these documents were used to run all experiments. To replicate the synthetic QA generation pipeline, you may choose any of the following options:

1. Collect your own documents and use these when calling the ai71 API for synthetic QA generation.
2. Download the FineWeb-10BT corpus (or a portion of it) here -- the sample-10BT dataset is about 27.6 GB: https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/main/sample/10BT. You may then replicate our procedure from the same corpus using the ai71 API synthetic QA generation.
3. Alternatively, you may choose to replicate the DataMorgana pipeline for synthetic QA generation, as described in the DataMorgana paper: ["Generating Diverse Q&A Benchmarks for RAG Evaluation with DataMorgana"](<https://arxiv.org/pdf/2501.12789>)

For the following steps, we assume a choice of option (2), which is the closest replication of our procedure.

## 1. Given a data store of the FineWeb-10BT corpus, and the DataMorgana tool, generate synthetic question-answer (QA) pairs

- Requirements: access to DataMorgana tool on ai71 website, ai71 API key. DataMorgana offers 10,000 credits to begin.
To generate synthetic QAs with randomly-selected document IDs:
- First, specify the desired configuration of your synthetic QAs to be generated in categorization_configs.py.
- Note: since we ran two rounds of generation for RQ1, we used two config files: using categorization_configs.py, and using EXTRA_categorization_configs.py
- Next, use the ipynb "Driver_Synthetic_QA_RAG_Benchmark.ipynb" to generate the synthetic QAs (with specified categorization configurations, sample sizes)

To generate synthetic QAs with fixed document IDs, you may specify the FineWeb-10BT document ID full uurn in the "document_ids" array field of the ai71 API request.

## 2. Continue on to generate answers using the RAG pipeline

- Navigate to /rag_system/generate/
- Then, to generate RAG answers for synthetic QAs from RQ1, RQ2, RQ3: Run main.py using gen_template.sh.
