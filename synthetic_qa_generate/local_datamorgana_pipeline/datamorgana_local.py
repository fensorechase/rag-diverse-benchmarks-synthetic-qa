"""
datamorgana_local.py
--------------------
A local replication of the DataMorgana synthetic QA generation pipeline
(Filice et al., 2025 — https://arxiv.org/abs/2501.12789).

Runs against any HuggingFace-compatible language model (e.g. Llama-3,
Mistral, Falcon-3) via the `transformers` library.  No ai71 account or
DataMorgana API key is needed.

Pipeline (mirrors the original paper, §3.2):
  1. Sample one category from every defined categorization, according
     to the configured probability weights.
  2. Draw a document from your local corpus.
  3. Build a prompt that injects the document + sampled category
     descriptions, then ask the LLM to produce k candidate QA pairs.
  4. Filter candidates: keep only pairs that (a) parse as valid JSON,
     (b) contain non-trivial text, and (c) are faithful to the document
     (lightweight keyword-overlap check, or optionally NLI-based).
  5. Sample one passing pair and emit it alongside its category labels.

Usage
-----
    python datamorgana_local.py \
        --corpus_dir  ./my_corpus_txt/    \
        --output      ./synthetic_qa.jsonl \
        --n_questions 200 \
        --model       meta-llama/Meta-Llama-3-8B-Instruct \
        --config      configs/default_config.json

See configs/default_config.json for the full categorization schema.
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1.  Category sampling
# ---------------------------------------------------------------------------

def sample_categories(categorizations: List[Dict]) -> Tuple[List[Dict], str]:
    """
    For each categorization, sample exactly one category according to the
    defined probabilities.

    Returns
    -------
    selected : list of {"categorization_name": ..., "category_name": ...,
                        "description": ...}
    combo_key : a human-readable string encoding the selected combination,
                useful as a dictionary key / log label.
    """
    selected = []
    combo_parts = []

    for cat in categorizations:
        cname = cat["categorization_name"]
        categories = cat["categories"]

        names = [c["name"] for c in categories]
        probs = [c.get("probability", 1.0 / len(categories)) for c in categories]

        # Normalise probabilities in case they don't sum to exactly 1.0
        total = sum(probs)
        probs = [p / total for p in probs]

        chosen = random.choices(categories, weights=probs, k=1)[0]
        selected.append(
            {
                "categorization_name": cname,
                "category_name": chosen["name"],
                "description": chosen["description"],
            }
        )
        combo_parts.append(f"{cname}={chosen['name']}")

    return selected, " | ".join(combo_parts)


# ---------------------------------------------------------------------------
# 2.  Prompt construction  (faithfully replicates the DataMorgana template)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
You are a user simulator that should generate {k} candidate questions for \
starting a conversation.

The {k} questions must be about facts discussed in the document you will now \
receive. When generating the questions, assume that the real users you must \
simulate, as well as the readers of the questions, do not have access to this \
document. Therefore, never refer to the author of the document or the document \
itself. Also, assume that whoever reads the questions will read each question \
independently. The {k} questions must be diverse and different from each other. \
Return ONLY the questions and answers — no preamble, no explanation. \
Write each pair on a new line in the following JSON format:
{{"question": "<question>", "answer": "<answer>"}}

### The generated questions should be about facts from the following document:

{document}

### Each of the generated questions must reflect a user with the following \
characteristics:
{user_constraints}
### Each of the generated questions must have the following characteristics:
{question_constraints}
"""


def build_prompt(
    document: str,
    question_categories: List[Dict],
    user_categories: List[Dict],
    k: int = 3,
) -> str:
    """
    Assemble the full generation prompt from the sampled categories and
    the source document text.

    Parameters
    ----------
    document          : raw text of the source document chunk
    question_categories : list of sampled question categorisation dicts
    user_categories     : list of sampled user categorisation dicts
    k                   : number of candidate QA pairs to request
    """
    user_lines = "\n".join(
        f"- They must be {c['description']}" for c in user_categories
    )
    question_lines = "\n".join(
        f"- It must be {c['description']}" for c in question_categories
    )

    return PROMPT_TEMPLATE.format(
        k=k,
        document=document.strip(),
        user_constraints=user_lines,
        question_constraints=question_lines,
    )


# ---------------------------------------------------------------------------
# 3.  QA filtering  (lightweight, no external model required by default)
# ---------------------------------------------------------------------------

def _keyword_overlap(answer: str, document: str, min_overlap: float = 0.10) -> bool:
    """
    Accept the answer if at least `min_overlap` fraction of its content
    words also appear in the source document (case-insensitive).
    This is a fast faithfulness proxy; use NLI filtering for higher quality.
    """
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "and", "but", "or", "so", "yet",
        "both", "either", "neither", "not", "no", "nor", "just", "that",
        "this", "these", "those", "it", "its", "i", "we", "you", "he", "she",
        "they", "what", "which", "who", "when", "where", "why", "how",
    }
    doc_words = set(re.findall(r"\b\w+\b", document.lower())) - stopwords
    ans_words = [w for w in re.findall(r"\b\w+\b", answer.lower()) if w not in stopwords]
    if not ans_words:
        return False
    overlap = sum(1 for w in ans_words if w in doc_words) / len(ans_words)
    return overlap >= min_overlap


def filter_candidates(
    raw_text: str,
    document: str,
    min_answer_len: int = 10,
    min_question_len: int = 8,
) -> List[Dict[str, str]]:
    """
    Parse raw model output and return QA pairs that pass quality checks.

    Accepts JSON lines in the format {"question": "...", "answer": "..."}
    Malformed lines are skipped gracefully.
    """
    valid = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip any markdown code fences the model might emit
        line = re.sub(r"^```(?:json)?", "", line).strip("`").strip()
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Try to extract JSON-like substring
            m = re.search(r'\{[^{}]*"question"[^{}]*"answer"[^{}]*\}', line)
            if not m:
                continue
            try:
                obj = json.loads(m.group())
            except json.JSONDecodeError:
                continue

        q = str(obj.get("question", "")).strip()
        a = str(obj.get("answer", "")).strip()

        if len(q) < min_question_len or len(a) < min_answer_len:
            continue
        if not _keyword_overlap(a, document):
            log.debug("Filtered out (low faithfulness): %s", q[:80])
            continue

        valid.append({"question": q, "answer": a})

    return valid


# ---------------------------------------------------------------------------
# 4.  Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(corpus_dir: str) -> List[Dict[str, str]]:
    """
    Load all .txt and .jsonl files from corpus_dir.

    For .txt  files : each file becomes one document.
    For .jsonl files: each line is expected to have at least a "text" field.
                      Optional "id" field is preserved.

    Returns a list of {"id": ..., "text": ...} dicts.
    """
    docs = []
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    for fp in sorted(corpus_path.iterdir()):
        if fp.suffix == ".txt":
            text = fp.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                docs.append({"id": fp.stem, "text": text})

        elif fp.suffix == ".jsonl":
            with open(fp, encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = obj.get("text") or obj.get("content") or obj.get("passage", "")
                    if text:
                        doc_id = obj.get("id") or obj.get("doc_id") or f"{fp.stem}_{i}"
                        docs.append({"id": str(doc_id), "text": str(text)})

    log.info("Loaded %d documents from %s", len(docs), corpus_dir)
    return docs


# ---------------------------------------------------------------------------
# 5.  HuggingFace model wrapper
# ---------------------------------------------------------------------------

class LocalLLM:
    """
    Thin wrapper around a HuggingFace AutoModelForCausalLM / pipeline.

    Supports any instruction-tuned model that follows a chat template
    (Llama-3, Mistral, Falcon-3, Qwen-2, etc.).
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        load_in_4bit: bool = False,
    ):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
        except ImportError:
            sys.exit(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            )

        log.info("Loading tokenizer from %s …", model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
            except ImportError:
                log.warning("bitsandbytes not installed; ignoring --load_in_4bit.")
        else:
            model_kwargs["torch_dtype"] = "auto"

        if device != "auto":
            model_kwargs["device_map"] = device

        log.info("Loading model …")
        self.pipe = pipeline(
            "text-generation",
            model=model_name_or_path,
            tokenizer=self.tokenizer,
            device_map=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            return_full_text=False,
            **{k: v for k, v in model_kwargs.items() if k not in ("device_map",)},
        )
        log.info("Model ready.")

    def generate(self, prompt: str) -> str:
        """Run the model and return the generated text."""
        # Use chat template if the tokenizer supports it, otherwise raw prompt
        if hasattr(self.tokenizer, "apply_chat_template") and \
                self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        outputs = self.pipe(formatted)
        return outputs[0]["generated_text"]


# ---------------------------------------------------------------------------
# 6.  Main generation loop
# ---------------------------------------------------------------------------

def generate_synthetic_qa(
    corpus: List[Dict[str, str]],
    question_categorizations: List[Dict],
    user_categorizations: List[Dict],
    llm: "LocalLLM",
    n_questions: int = 200,
    k_candidates: int = 3,
    max_retries: int = 3,
) -> List[Dict]:
    """
    Generate `n_questions` synthetic QA pairs from the local corpus.

    Each question is generated from a randomly sampled document and a
    randomly sampled combination of category values.

    Returns a list of result dicts, each containing:
        question, answer, document_id, question_categories, user_categories,
        combo_key
    """
    results = []
    attempts = 0
    max_total_attempts = n_questions * (max_retries + 2)

    while len(results) < n_questions and attempts < max_total_attempts:
        attempts += 1

        # --- Sample document ---
        doc = random.choice(corpus)
        document_text = doc["text"]
        # Truncate very long documents to avoid context-length issues
        document_text = document_text[:4000]

        # --- Sample categories ---
        q_selected, q_combo = sample_categories(question_categorizations)
        u_selected, u_combo = sample_categories(user_categorizations)
        combo_key = f"{u_combo} | {q_combo}"

        # --- Build prompt ---
        prompt = build_prompt(
            document=document_text,
            question_categories=q_selected,
            user_categories=u_selected,
            k=k_candidates,
        )

        # --- Generate ---
        try:
            raw = llm.generate(prompt)
        except Exception as exc:
            log.warning("Generation error: %s — retrying.", exc)
            continue

        # --- Filter ---
        candidates = filter_candidates(raw, document_text)
        if not candidates:
            log.debug("No valid candidates for combo: %s", combo_key)
            continue

        # Pick one candidate at random (mirrors the DataMorgana filtering step)
        chosen = random.choice(candidates)

        results.append(
            {
                "question": chosen["question"],
                "answer": chosen["answer"],
                "document_id": doc["id"],
                "question_categories": q_selected,
                "user_categories": u_selected,
                "combo_key": combo_key,
            }
        )

        if len(results) % 10 == 0:
            log.info("Generated %d / %d questions …", len(results), n_questions)

    if len(results) < n_questions:
        log.warning(
            "Only generated %d / %d questions after %d attempts.",
            len(results), n_questions, attempts,
        )

    return results


# ---------------------------------------------------------------------------
# 7.  CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local DataMorgana-style synthetic QA generation"
    )
    p.add_argument(
        "--corpus_dir",
        default="./sample_corpus",
        help="Directory containing .txt or .jsonl document files",
    )
    p.add_argument(
        "--config",
        default="configs/default_config.json",
        help="JSON file specifying question and user categorizations",
    )
    p.add_argument(
        "--output",
        default="./synthetic_qa.jsonl",
        help="Output JSONL file path",
    )
    p.add_argument(
        "--n_questions",
        type=int,
        default=200,
        help="Number of synthetic QA pairs to generate",
    )
    p.add_argument(
        "--k_candidates",
        type=int,
        default=3,
        help="Number of candidate QA pairs to generate per document (then filter to 1)",
    )
    p.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="HuggingFace model name or local path",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Device map for model loading (auto, cpu, cuda, cuda:0, …)",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    p.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )
    p.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantisation (requires bitsandbytes)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config file not found: {args.config}")
    with open(config_path) as fh:
        config = json.load(fh)

    question_cats = config.get("question_categorizations", [])
    user_cats = config.get("user_categorizations", [])

    if not question_cats:
        sys.exit("No question_categorizations found in config.")
    if not user_cats:
        sys.exit("No user_categorizations found in config.")

    # Load corpus
    corpus = load_corpus(args.corpus_dir)
    if not corpus:
        sys.exit("No documents loaded from corpus_dir. Check the path and file formats.")

    # Load model
    llm = LocalLLM(
        model_name_or_path=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        load_in_4bit=args.load_in_4bit,
    )

    # Generate
    results = generate_synthetic_qa(
        corpus=corpus,
        question_categorizations=question_cats,
        user_categorizations=user_cats,
        llm=llm,
        n_questions=args.n_questions,
        k_candidates=args.k_candidates,
    )

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for item in results:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    log.info("Saved %d QA pairs to %s", len(results), out_path)


if __name__ == "__main__":
    main()
