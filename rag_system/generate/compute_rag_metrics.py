"""
compute_rag_metrics.py
Compute retrieval and generation metrics for RQ1, RQ2, and RQ3 results
Updated for new hierarchical 2→4→8 structure
"""
import json
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

# Load semantic similarity model
semantic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL file - handles both line-delimited and array format"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        # Check if it's a JSON array
        if content.startswith('['):
            return json.loads(content)
        
        # Otherwise treat as JSONL (one JSON object per line)
        lines = content.split('\n')
        results = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line {i+1}: {e}")
                continue
        return results

def compute_rouge(reference: str, generated: str) -> Dict[str, float]:
    """Compute ROUGE scores"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def compute_bleu(reference: str, generated: str) -> float:
    """Compute BLEU score"""
    reference_tokens = reference.lower().split()
    generated_tokens = generated.lower().split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothie)

def compute_cosine_similarity(reference: str, generated: str) -> float:
    """Compute cosine similarity using MiniLM embeddings"""
    embeddings = semantic_model.encode([reference, generated])
    cos_sim = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return float(cos_sim)

def is_refusal(answer: str) -> bool:
    """Check if the answer is a refusal"""
    refusal_patterns = [
        r"i don't have enough information",
        r"i cannot answer",
        r"i don't know",
        r"insufficient information",
        r"not enough information",
        r"cannot determine",
        r"unable to answer"
    ]
    answer_lower = answer.lower()
    return any(re.search(pattern, answer_lower) for pattern in refusal_patterns)

def compute_generation_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute generation metrics for a set of results"""
    rouge1_scores = []
    rougeL_scores = []
    bleu_scores = []
    cosine_scores = []
    refusal_count = 0
    
    for item in results:
        reference = item.get('reference_answer', item.get('answer', ''))
        generated = item.get('generated_answer', '')
        
        if not reference or not generated:
            continue
        
        # ROUGE
        rouge = compute_rouge(reference, generated)
        rouge1_scores.append(rouge['rouge1'])
        rougeL_scores.append(rouge['rougeL'])
        
        # BLEU
        bleu = compute_bleu(reference, generated)
        bleu_scores.append(bleu)
        
        # Cosine similarity
        cos_sim = compute_cosine_similarity(reference, generated)
        cosine_scores.append(cos_sim)
        
        # Refusal
        if is_refusal(generated):
            refusal_count += 1
    
    return {
        'rouge1': np.mean(rouge1_scores) if rouge1_scores else 0.0,
        'rouge1_std': np.std(rouge1_scores) if rouge1_scores else 0.0,
        'rougeL': np.mean(rougeL_scores) if rougeL_scores else 0.0,
        'bleu': np.mean(bleu_scores) if bleu_scores else 0.0,
        'cosine_sim': np.mean(cosine_scores) if cosine_scores else 0.0,
        'cosine_sim_std': np.std(cosine_scores) if cosine_scores else 0.0,
        'refusal_rate': (refusal_count / len(results) * 100) if results else 0.0,
        'n_samples': len(results)
    }

def compute_retrieval_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute retrieval metrics (MAP, nDCG@10, Recall@10)
    Uses retrieved_document_ids field (now in UUID format matching ground truth)
    """
    map_scores = []
    ndcg_scores = []
    recall_at_10 = []
    
    for item in results:
        # Ground truth document IDs
        gold_docs = set(item.get('document_ids', []))
        if not gold_docs:
            continue
        
        # Retrieved document IDs (now in same format as ground truth)
        retrieved = item.get('retrieved_document_ids', [])[:10]
        
        if not retrieved:
            map_scores.append(0.0)
            ndcg_scores.append(0.0)
            recall_at_10.append(0.0)
            continue
        
        # Average Precision
        relevant_retrieved = []
        for i, doc in enumerate(retrieved):
            if doc in gold_docs:
                relevant_retrieved.append(i + 1)
        
        if relevant_retrieved:
            precisions = [len([r for r in relevant_retrieved if r <= pos]) / pos 
                         for pos in relevant_retrieved]
            avg_precision = sum(precisions) / len(gold_docs)
            map_scores.append(avg_precision)
        else:
            map_scores.append(0.0)
        
        # nDCG@10
        dcg = 0.0
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(gold_docs), 10))])
        for i, doc in enumerate(retrieved):
            if doc in gold_docs:
                dcg += 1.0 / np.log2(i + 2)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
        
        # Recall@10
        found = sum(1 for doc in retrieved if doc in gold_docs)
        recall = found / len(gold_docs) if gold_docs else 0.0
        recall_at_10.append(recall)
    
    return {
        'map': np.mean(map_scores) if map_scores else 0.0,
        'ndcg@10': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        'recall@10': np.mean(recall_at_10) if recall_at_10 else 0.0,
        'n_samples': len(map_scores)
    }

def analyze_rq1(results_file: str = './results/rq1_total_fixed.jsonl'):
    """Analyze RQ1 results by categorization and granularity"""
    print("\n" + "="*80)
    print("RQ1: GRANULARITY ANALYSIS - RESULTS")
    print("="*80)
    
    results = load_jsonl(results_file)
    print(f"Loaded {len(results)} results")
    
    # Group by categorization and granularity
    grouped = defaultdict(lambda: defaultdict(list))
    
    for item in results:
        cat = item.get('categorization', '')
        gran = item.get('granularity', '')
        grouped[cat][gran].append(item)
    
    all_results = {}
    
    for cat in ['question_complexity', 'question_answertype', 'question_linguisticvariation']:
        print(f"\n{cat.upper().replace('_', ' ')}:")
        all_results[cat] = {}
        
        for gran in ['coarse', 'medium', 'fine']:
            items = grouped[cat][gran]
            if not items:
                continue
            
            print(f"\n  {gran.capitalize()} ({len(items)} questions):")
            
            # Retrieval metrics
            retrieval = compute_retrieval_metrics(items)
            print(f"    MAP: {retrieval['map']:.3f}")
            print(f"    nDCG@10: {retrieval['ndcg@10']:.3f}")
            print(f"    Recall@10: {retrieval['recall@10']:.3f}")
            
            # Generation metrics
            generation = compute_generation_metrics(items)
            print(f"    ROUGE-1: {generation['rouge1']:.3f}")
            print(f"    BLEU: {generation['bleu']:.3f}")
            print(f"    Cosine Sim: {generation['cosine_sim']:.3f} (±{generation['cosine_sim_std']:.3f})")
            print(f"    Refusal Rate: {generation['refusal_rate']:.1f}%")
            
            all_results[cat][gran] = {
                'retrieval': retrieval,
                'generation': generation
            }
        
        # Compute discriminative power (std dev of cosine similarity across categories)
        print(f"\n  Discriminative Power (Cosine Sim Std Dev across categories):")
        for gran in ['coarse', 'medium', 'fine']:
            items = grouped[cat][gran]
            if not items:
                continue
            
            # Group by individual category
            by_category = defaultdict(list)
            for item in items:
                for qcat in item.get('question_categories', []):
                    if qcat['categorization_name'] == cat:
                        by_category[qcat['category_name']].append(item)
            
            # Compute mean cosine sim per category
            category_means = []
            for cat_name, cat_items in by_category.items():
                gen_metrics = compute_generation_metrics(cat_items)
                category_means.append(gen_metrics['cosine_sim'])
                print(f"    {gran}/{cat_name}: {gen_metrics['cosine_sim']:.3f} (n={len(cat_items)})")
            
            std_dev = np.std(category_means) if len(category_means) > 1 else 0.0
            print(f"    → Std Dev across categories: {std_dev:.3f}")
            all_results[cat][gran]['discriminative_power'] = std_dev
    
    return all_results

def analyze_rq2(results_file: str = './results/rq2_results_fixed.jsonl'):
    """Analyze RQ2 results by categorization set"""
    print("\n" + "="*80)
    print("RQ2: COMPLEMENTARITY ANALYSIS - RESULTS")
    print("="*80)
    
    results = load_jsonl(results_file)
    print(f"Loaded {len(results)} results")
    
    # Group by categorization set
    grouped = defaultdict(list)
    
    for item in results:
        cat_set = item.get('categorization_set', '')
        grouped[cat_set].append(item)
    
    all_results = {}
    
    set_names = {
        'complexity': 'Set 1: Complexity',
        'answertype': 'Set 2: Answer Type',
        'vocabulary': 'Set 3: Vocabulary',
        'phrasing': 'Set 4: Phrasing',
        'expertise': 'Set 5: User Expertise'
    }
    
    for cat_set, display_name in set_names.items():
        items = grouped[cat_set]
        if not items:
            continue
        
        print(f"\n{display_name} ({len(items)} questions):")
        
        # Retrieval metrics
        retrieval = compute_retrieval_metrics(items)
        print(f"  MAP: {retrieval['map']:.3f}")
        print(f"  nDCG@10: {retrieval['ndcg@10']:.3f}")
        print(f"  Recall@10: {retrieval['recall@10']:.3f}")
        
        # Generation metrics
        generation = compute_generation_metrics(items)
        print(f"  ROUGE-1: {generation['rouge1']:.3f}")
        print(f"  BLEU: {generation['bleu']:.3f}")
        print(f"  Cosine Sim: {generation['cosine_sim']:.3f} (±{generation['cosine_sim_std']:.3f})")
        print(f"  Refusal Rate: {generation['refusal_rate']:.1f}%")
        
        # Compute performance range
        by_category = defaultdict(list)
        for item in items:
            for qcat in item.get('question_categories', []):
                by_category[qcat['category_name']].append(item)
        
        category_means = []
        for cat_name, cat_items in by_category.items():
            gen_metrics = compute_generation_metrics(cat_items)
            category_means.append(gen_metrics['cosine_sim'])
        
        perf_range = max(category_means) - min(category_means) if category_means else 0.0
        print(f"  Performance Range: {perf_range:.3f}")
        
        all_results[cat_set] = {
            'retrieval': retrieval,
            'generation': generation,
            'performance_range': perf_range
        }
    
    return all_results

def analyze_rq3(results_file: str = './results/rq3_results_fixed.jsonl'):
    """Analyze RQ3 2x2 factorial results"""
    print("\n" + "="*80)
    print("RQ3: INTERACTION ANALYSIS - RESULTS")
    print("="*80)
    
    results = load_jsonl(results_file)
    print(f"Loaded {len(results)} results")
    
    # Group by 2x2 factorial
    factorial = defaultdict(list)
    
    for item in results:
        vocab_cat = None
        complexity_cat = None
        
        for qcat in item.get('question_categories', []):
            if qcat['categorization_name'] == 'question_linguisticvariation':
                vocab_cat = qcat['category_name']
            elif qcat['categorization_name'] == 'question_complexity':
                complexity_cat = qcat['category_name']
        
        if vocab_cat and complexity_cat:
            key = f"{vocab_cat}_{complexity_cat}"
            factorial[key].append(item)
    
    print("\n2×2 Factorial Results:")
    all_results = {}
    
    for key in sorted(factorial.keys()):
        items = factorial[key]
        # Split key properly (handle multi-word categories)
        parts = key.rsplit('_', 1)
        if len(parts) != 2:
            continue
        vocab, complexity = parts
        
        print(f"\n  {vocab.replace('_', ' ').title()} × {complexity.title()} ({len(items)} questions):")
        
        # Retrieval metrics
        retrieval = compute_retrieval_metrics(items)
        print(f"    MAP: {retrieval['map']:.3f}")
        print(f"    nDCG@10: {retrieval['ndcg@10']:.3f}")
        
        # Generation metrics
        generation = compute_generation_metrics(items)
        print(f"    ROUGE-1: {generation['rouge1']:.3f}")
        print(f"    Cosine Sim: {generation['cosine_sim']:.3f}")
        print(f"    Refusal Rate: {generation['refusal_rate']:.1f}%")
        
        all_results[key] = {
            'retrieval': retrieval,
            'generation': generation
        }
    
    return all_results

def generate_rq1_latex_table(rq1_results):
    """Generate compact LaTeX table for RQ1 with abbreviations"""
    
    # Category abbreviations
    abbrev = {
        'question_complexity': 'QC',
        'question_answertype': 'AT',
        'question_linguisticvariation': 'LV'
    }
    
    latex = r"""
\begin{table*}[t]
\centering
\caption{RQ1: RAG performance across categorical granularity levels. \textbf{Abbreviations:} QC=Question Complexity, AT=Answer Type, LV=Linguistic Variation, C=Coarse (2 cat.), M=Medium (4 cat.), F=Fine (8 cat.), MAP=Mean Average Precision, CS=Cosine Similarity, Ref=Refusal Rate, DiscPow=Discriminative Power (std dev of CS across categories).}
\label{tab:rq1_results}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccccc}
\toprule
\textbf{Cat.} & \textbf{Gran.} & \textbf{MAP} & \textbf{nDCG@10} & \textbf{R-1} & \textbf{CS} & \textbf{Ref\%} & \textbf{DiscPow} \\
\midrule
"""
    
    for cat_key in ['question_complexity', 'question_answertype', 'question_linguisticvariation']:
        cat_abbr = abbrev[cat_key]
        
        for i, gran in enumerate(['coarse', 'medium', 'fine']):
            if gran not in rq1_results[cat_key]:
                continue
            
            r = rq1_results[cat_key][gran]
            ret = r['retrieval']
            gen = r['generation']
            disc_pow = r.get('discriminative_power', 0.0)
            
            gran_abbr = gran[0].upper()  # C, M, F
            
            # Only show category name on first row
            if i == 0:
                latex += f"{cat_abbr} & {gran_abbr} & "
            else:
                latex += f"& {gran_abbr} & "
            
            latex += f"{ret['map']:.3f} & {ret['ndcg@10']:.3f} & {gen['rouge1']:.3f} & "
            latex += f"{gen['cosine_sim']:.3f} & {gen['refusal_rate']:.1f} & {disc_pow:.3f} \\\\\n"
        
        if cat_key != 'question_linguisticvariation':
            latex += "\\midrule\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    
    return latex

def main():
    """Run all analyses"""
    print("="*80)
    print("RAG RESULTS ANALYSIS")
    print("="*80)
    
    # Analyze RQ1
    rq1_results = analyze_rq1('./results/rq1_total_fixed.jsonl')
    
    # Analyze RQ2
    rq2_results = analyze_rq2('./results/rq2_results_fixed.jsonl')
    
    # Analyze RQ3
    rq3_results = analyze_rq3('./results/rq3_results_fixed.jsonl')
    
    # Generate LaTeX table
    rq1_latex = generate_rq1_latex_table(rq1_results)
    
    # Save all results
    with open('rag_analysis_results.json', 'w') as f:
        json.dump({
            'rq1': rq1_results,
            'rq2': rq2_results,
            'rq3': rq3_results
        }, f, indent=2, default=str)
    
    # Save LaTeX table
    with open('rq1_table.tex', 'w') as f:
        f.write(rq1_latex)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("Results saved to: rag_analysis_results.json")
    print("LaTeX table saved to: rq1_table.tex")
    print("="*80)
    
    print("\n" + "="*80)
    print("RQ1 LATEX TABLE:")
    print("="*80)
    print(rq1_latex)

if __name__ == "__main__":
    main()