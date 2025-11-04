"""
compute_rag_metrics.py
Compute retrieval and generation metrics for RQ1, RQ2, and RQ3 results
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
            # Fallback: try old field name if new one doesn't exist
            retrieved = item.get('retrieved_docnos', [])[:10]
        
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

def analyze_rq1(results_file: str = 'rq1_results.jsonl'):
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
            print(f"    ROUGE-L: {generation['rougeL']:.3f}")
            print(f"    BLEU: {generation['bleu']:.3f}")
            print(f"    Cosine Sim: {generation['cosine_sim']:.3f} (±{generation['cosine_sim_std']:.3f})")
            print(f"    Refusal Rate: {generation['refusal_rate']:.1f}%")
            
            all_results[cat][gran] = {
                'retrieval': retrieval,
                'generation': generation
            }
        
        # Compute std dev of cosine similarity across categories within granularity
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
                print(f"    {gran}/{cat_name}: {gen_metrics['cosine_sim']:.3f}")
            
            std_dev = np.std(category_means) if len(category_means) > 1 else 0.0
            print(f"    → Std Dev across categories: {std_dev:.3f}")
            all_results[cat][gran]['discriminative_power'] = std_dev
    
    return all_results

def analyze_rq2(results_file: str = 'rq2_results.jsonl'):
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
        
        # Compute performance range (max - min cosine sim across categories)
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

def analyze_rq3(results_file: str = 'rq3_results.jsonl'):
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
    
    print("\n2x2 Factorial Results:")
    all_results = {}
    
    for key in sorted(factorial.keys()):
        items = factorial[key]
        #vocab, complexity = key.split('_')
        parts = key.rsplit('_', 1)  # Split from right, only once
        if len(parts) == 2:
                    vocab, complexity = parts
        else:
            continue  # Skip malformed keys

        #print(f"\n  {vocab.replace('_', ' ').title()} x {complexity.title()} ({len(items)} questions):")
        
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

def generate_combined_latex_table(rq1_results, rq2_results):
    """Generate combined LaTeX table for RQ1 and RQ2"""
    print("\n" + "="*80)
    print("COMBINED LATEX TABLE (RQ1 + RQ2)")
    print("="*80)
    
    latex = r"""
\begin{table*}[t]
\centering
\caption{RAG system performance across categorical granularities (RQ1) and categorization sets (RQ2). Std Dev measures discriminative power in RQ1.}
\label{tab:rag_results_combined}
\scriptsize
\begin{tabular}{llcccccc}
\toprule
\textbf{Categorization} & \textbf{Granularity/Set} & \textbf{MAP} & \textbf{nDCG@10} & \textbf{ROUGE-1} & \textbf{Cos. Sim.} & \textbf{Refusal \%} & \textbf{Std Dev} \\
\midrule
\multicolumn{8}{l}{\textit{RQ1: Granularity Analysis}} \\
\midrule
"""
    
    # RQ1 sections
    cat_display = {
        'question_complexity': 'Question Complexity',
        'question_answertype': 'Answer Type',
        'question_linguisticvariation': 'Linguistic Variation'
    }
    
    for cat in ['question_complexity', 'question_answertype', 'question_linguisticvariation']:
        latex += f"\\multirow{{3}}{{*}}{{{cat_display[cat]}}}\n"
        for i, gran in enumerate(['coarse', 'medium', 'fine']):
            if gran not in rq1_results[cat]:
                continue
            r = rq1_results[cat][gran]
            ret = r['retrieval']
            gen = r['generation']
            std_dev = r.get('discriminative_power', 0.0)
            
            if i == 0:
                latex += f"& {gran.capitalize()} & {ret['map']:.3f} & {ret['ndcg@10']:.3f} & {gen['rouge1']:.3f} & {gen['cosine_sim']:.3f} & {gen['refusal_rate']:.1f} & {std_dev:.3f} \\\\\n"
            else:
                latex += f"& {gran.capitalize()} & {ret['map']:.3f} & {ret['ndcg@10']:.3f} & {gen['rouge1']:.3f} & {gen['cosine_sim']:.3f} & {gen['refusal_rate']:.1f} & {std_dev:.3f} \\\\\n"
        latex += "\\midrule\n"
    
    # RQ2 section
    latex += r"""
\multicolumn{8}{l}{\textit{RQ2: Complementarity Analysis}} \\
\midrule
"""
    
    set_names = {
        'complexity': 'Complexity',
        'answertype': 'Answer Type',
        'vocabulary': 'Vocabulary',
        'phrasing': 'Phrasing',
        'expertise': 'User Expertise'
    }
    
    for cat_set in ['complexity', 'answertype', 'vocabulary', 'phrasing', 'expertise']:
        if cat_set not in rq2_results:
            continue
        r = rq2_results[cat_set]
        ret = r['retrieval']
        gen = r['generation']
        perf_range = r.get('performance_range', 0.0)
        
        latex += f"\\multicolumn{{2}}{{l}}{{{set_names[cat_set]}}} & {ret['map']:.3f} & {ret['ndcg@10']:.3f} & {gen['rouge1']:.3f} & {gen['cosine_sim']:.3f} & {gen['refusal_rate']:.1f} & {perf_range:.3f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\scriptsize
\item \textbf{Std Dev} (RQ1): Standard deviation of cosine similarity across categories within each granularity level. Higher values indicate better discriminative power.
\item \textbf{Std Dev} (RQ2): Performance range (max - min cosine similarity) across categories within each set.
\end{tablenotes}
\end{table*}
"""
    
    print(latex)
    return latex

def main():
    """Run all analyses"""
    print("="*80)
    print("RAG RESULTS ANALYSIS")
    print("="*80)
    
    # Analyze RQ1
    rq1_results = analyze_rq1('./results/rq1_results.jsonl')
    
    # Analyze RQ2
    rq2_results = analyze_rq2('./results/rq2_results.jsonl')
    
    # Analyze RQ3
    rq3_results = analyze_rq3('./results/rq3_results.jsonl')
    
    # Generate combined table
    combined_latex = generate_combined_latex_table(rq1_results, rq2_results)
    
    # Save all results
    with open('./results/rag_analysis_results.json', 'w') as f:
        json.dump({
            'rq1': rq1_results,
            'rq2': rq2_results,
            'rq3': rq3_results
        }, f, indent=2, default=str)
    
    # Save LaTeX table
    with open('./results/combined_table.tex', 'w') as f:
        f.write(combined_latex)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("Results saved to: rag_analysis_results.json")
    print("LaTeX table saved to: combined_table.tex")
    print("="*80)

if __name__ == "__main__":
    main()