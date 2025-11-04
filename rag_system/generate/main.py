"""
main.py
Lean RAG QA script using local Falcon-10B-Instruct or ai71 Falcon (API key)
Only uses local BM25 PyTerrier sparse index for retrieval.
"""
import os
import sys
import re
import argparse
import torch
import json
import time
import multiprocessing
import math
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from typing import Dict, Any, List
from config import (
    LLM_MODEL_ID,
    RESULTS_DIR,
    BM25_INDEX_PATH,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    DO_SAMPLE
)
from utils import setup_logger, log_execution_time, save_results
from retriever_utils import get_sparse_index, cleanup


import pyterrier as pt
if not pt.started():
    pt.init()


# Optional: ai71 Falcon API support
try:
    from ai71 import FalconAPIClient  # You must provide this module if using ai71
    AI71_AVAILABLE = True
except ImportError:
    AI71_AVAILABLE = False

# HuggingFace Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = setup_logger("main")

def load_questions(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r") as f:
        if input_path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        else:
            return json.load(f)

def get_bm25_retriever():
    index = get_sparse_index(BM25_INDEX_PATH)
    import pyterrier as pt
    bm25 = pt.terrier.Retriever(index, wmodel="BM25")
    return bm25, index

def extract_generated_answer(text):
    # Split on last occurrence of "Answer:"
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip()
    return text.strip()

@log_execution_time
def retrieve_passages(query: str, bm25, index, top_k: int = 10) -> List[Dict[str, Any]]:
    import pandas as pd
    query_clean = cleanup(query)
    df = pd.DataFrame([{"qid": 0, "query": query_clean}])
    results = bm25.transform(df)
    passages = []
    docnos = results.head(top_k)["docno"].tolist()
    
    # Fetch text, source_file, and url (UUID is embedded in one of these)
    text_df = pt.text.get_text(index, ["text", "source_file", "url"])(pd.DataFrame({"docno": docnos}))
    
    for _, row in text_df.iterrows():
        docno = row["docno"]
        text = row.get("text", "")
        source_file = row.get("source_file", "")
        url = row.get("url", "")
        
        # Extract UUID from source_file or url
        # Pattern: <urn:uuid:XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX>
        uuid_pattern = r'urn:uuid:([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})'
        
        uuid = None
        # Try source_file first
        if source_file:
            match = re.search(uuid_pattern, source_file, re.IGNORECASE)
            if match:
                uuid = match.group(1)
        
        # If not found, try url
        if not uuid and url:
            match = re.search(uuid_pattern, url, re.IGNORECASE)
            if match:
                uuid = match.group(1)
        
        passages.append({
            "text": text,
            "docno": docno,
            "uuid": uuid if uuid else ""  # Will be formatted later
        })
    
    return passages

@log_execution_time
def generate_answer_falcon_local(question: str, passages: List[Dict[str, Any]], model, tokenizer) -> str:
    context = "\n".join([p["text"] for p in passages])
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Ensure tensors are on the correct device
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P, do_sample=DO_SAMPLE)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip()

@log_execution_time
def generate_answer_ai71(question: str, passages: List[Dict[str, Any]], api_client) -> str:
    context = "\n".join([p["text"] for p in passages])
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = api_client.generate(prompt)
    return response.strip()


def truncate_context_for_prompt(passages: List[Dict[str, Any]], 
                                question: str, 
                                tokenizer,
                                max_prompt_tokens: int = 900) -> str:
    """
    Intelligently truncate retrieved passages to fit token budget.
    Leaves room for question and Answer: prompt.
    """
    # Reserve tokens for question and prompt structure
    question_tokens = len(tokenizer.encode(question))
    prompt_overhead = 20  # For "Context:\n", "\n\nQuestion:", "\nAnswer:"
    available_tokens = max_prompt_tokens - question_tokens - prompt_overhead
    
    if available_tokens < 100:
        # Question is too long, truncate it
        question = tokenizer.decode(tokenizer.encode(question)[:100])
        available_tokens = max_prompt_tokens - 100 - prompt_overhead
    
    # Tokenize all passages
    passage_tokens = [tokenizer.encode(p["text"], add_special_tokens=False) 
                     for p in passages]
    total_tokens = sum(len(p) for p in passage_tokens)
    
    # If under budget, use all passages
    if total_tokens <= available_tokens:
        return "\n".join([p["text"] for p in passages])
    
    # Otherwise, proportionally truncate each passage
    truncated_passages = []
    tokens_per_passage = available_tokens // len(passages)
    
    for p, tokens in zip(passages, passage_tokens):
        if len(tokens) <= tokens_per_passage:
            truncated_passages.append(p["text"])
        else:
            # Truncate but keep complete sentences
            truncated = tokenizer.decode(tokens[:tokens_per_passage])
            # Try to end at last complete sentence
            last_period = truncated.rfind('.')
            if last_period > tokens_per_passage * 0.5:  # At least 50% remains
                truncated = truncated[:last_period + 1]
            truncated_passages.append(truncated + "...")
    
    return "\n".join(truncated_passages)


def batch_iterable(iterable, batch_size):
    """Yield successive batches from iterable."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

def retrieve_passages_worker(args):
    """Worker for multiprocessing retrieval."""
    question, bm25, index, top_k = args
    return retrieve_passages(question, bm25, index, top_k)

def generate_answers_batch(questions, passages_batch, model, tokenizer):
    """Batch Falcon answer generation with robust stopping."""
    prompts = []
    for q, passages in zip(questions, passages_batch):
        context = truncate_context_for_prompt(passages, q, tokenizer, max_prompt_tokens=900)
        prompts.append(f"Context:\n{context}\n\nQuestion: {q}\nAnswer:")
    
    # Tokenize prompts
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=1024
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,  # Increased from 512 - try shorter first
            min_new_tokens=20,   # Ensure at least 20 tokens generated
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            early_stopping=False  # Don't stop early
        )
    
    # Decode full outputs
    full_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Extract answers
    answers = []
    for i, full_text in enumerate(full_texts):
        # Remove the prompt part
        original_prompt = prompts[i]
        
        if "Answer:" in full_text:
            # Split on the last "Answer:" occurrence
            parts = full_text.split("Answer:")
            answer = parts[-1].strip()
        else:
            # Fallback: try to remove prompt
            if full_text.startswith(original_prompt):
                answer = full_text[len(original_prompt):].strip()
            else:
                answer = full_text.strip()
        
        # Clean artifacts
        answer = answer.replace("<|assistant|>", "").strip()
        answer = answer.replace("<|endoftext|>", "").strip()
        answer = answer.replace("<|end|>", "").strip()
        
        # If answer is suspiciously short, flag it
        if len(answer.split()) < 5:
            logger.warning(f"Short answer generated for Q{i}: '{answer}' (prompt length: {len(original_prompt)})")
        
        answers.append(answer)
    
    return answers



def main():
    parser = argparse.ArgumentParser(description="Lean RAG QA with Falcon-10B-Instruct and BM25")
    parser.add_argument("--input", required=True, help="Path to input questions (.json or .jsonl)")
    parser.add_argument("--output", required=True, help="Path to output results (.json)")
    parser.add_argument("--generator", choices=["local", "ai71"], default="local", help="Which Falcon generator to use")
    parser.add_argument("--ai71_api_key", type=str, default=None, help="API key for ai71 Falcon (if using ai71)")
    parser.add_argument("--top_k", type=int, default=10, help="Number of passages to retrieve")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for Falcon inference")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of CPU workers for retrieval")
    args = parser.parse_args()

    questions = load_questions(args.input)
    logger.info(f"Loaded {len(questions)} questions from {args.input}")

    bm25, index = get_bm25_retriever()

    if args.generator == "local":
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, token=os.environ["HF_TOKEN"])
        
        # Fix padding for decoder-only model
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID, token=os.environ["HF_TOKEN"],
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.eval()
    elif args.generator == "ai71":
        if not AI71_AVAILABLE:
            raise RuntimeError("ai71 Falcon API client not available. Please install ai71_falcon.")
        api_client = FalconAPIClient(api_key=args.ai71_api_key)
    else:
        raise ValueError("Invalid generator option.")

    results = []
    batch_size = args.batch_size

    pbar = tqdm(total=len(questions), desc="Processing questions", unit="q")
    for batch_start in range(0, len(questions), batch_size):
        batch_questions = questions[batch_start:batch_start + batch_size]
        batch_texts = [q.get("question", "") for q in batch_questions]

        t_batch_start = time.time()
        # Sequential retrieval (thread-safe)
        batch_passages = [retrieve_passages(q, bm25, index, args.top_k) for q in batch_texts]

        if args.generator == "local":
            batch_answers = generate_answers_batch(batch_texts, batch_passages, model, tokenizer)
        else:
            batch_answers = [generate_answer_ai71(q, p, api_client) for q, p in zip(batch_texts, batch_passages)]

        for i, (item, answer, passages) in enumerate(zip(batch_questions, batch_answers, batch_passages)):
            result = dict(item)
            result["generated_answer"] = answer
            result["reference_answer"] = item.get("answer", "")
            result["retrieved_docnos"] = [p["docno"] for p in passages]
            #result["retrieved_passages"] = passages
            result["item_index"] = batch_start + i

            # Extract unique document UUIDs
            retrieved_uuids = []
            seen = set()
            for p in passages:
                uuid = p.get("uuid", "")
                if uuid and uuid not in seen:
                    # Format to match ground truth: <urn:uuid:...>
                    formatted_uuid = f"<urn:uuid:{uuid}>"
                    retrieved_uuids.append(formatted_uuid)
                    seen.add(uuid)
            result["retrieved_document_ids"] = retrieved_uuids
            results.append(result)

            # Print progress every 100 questions
            if (batch_start + i + 1) % 100 == 0:
                logger.info(f"Q{batch_start + i + 1}: {item.get('question','')[:80]}")
                logger.info(f"Answer: {answer[:120]}")
                logger.info(f"Time for last batch: {time.time() - t_batch_start:.2f}s")

        pbar.update(len(batch_questions))

        if len(results) % 100 == 0 or batch_start + batch_size >= len(questions):
            save_results(results, args.output, RESULTS_DIR)

    pbar.close()
    logger.info(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
