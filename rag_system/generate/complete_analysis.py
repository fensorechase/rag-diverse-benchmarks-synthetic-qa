# complete_analysis.py
"""
Complete analysis for RQ1, RQ2, RQ3 with diversity metrics
Includes: Performance metrics, discriminative power, diversity scores, baseline comparison, MI analysis
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Any
import re
from tqdm import tqdm
import random
from scipy import stats

# NLP libraries
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy

random.seed(42)
np.random.seed(42)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load JSONL file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            return json.loads(content)
        return [json.loads(line) for line in content.split('\n') if line.strip()]

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

def compute_cosine_similarity(reference: str, generated: str, model) -> float:
    """Compute cosine similarity using embeddings"""
    embeddings = model.encode([reference, generated])
    cos_sim = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return float(cos_sim)

def is_refusal(answer: str) -> bool:
    """Check if answer is a refusal"""
    refusal_patterns = [
        r"i don't have enough information",
        r"i cannot answer",
        r"i don't know",
        r"insufficient information",
        r"not enough information",
        r"cannot determine",
        r"unable to answer",
        r"not enough information provided in the passage"
        ]
    answer_lower = answer.lower()
    return any(re.search(pattern, answer_lower) for pattern in refusal_patterns)

# ============================================================================
# DIVERSITY METRICS (from Shaib et al., 2024 / DataMorgana)
# ============================================================================

def compute_ndg_score(questions: List[str], max_n: int = 4) -> float:
    """
    N-Gram Diversity (NDG) Score
    Ratio of unique n-grams to total n-grams
    """
    total_ngrams = 0
    unique_ngrams = set()
    
    for n in range(1, max_n + 1):
        for question in questions:
            tokens = question.lower().split()
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            total_ngrams += len(ngrams)
            unique_ngrams.update(ngrams)
    
    return len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0.0

def compute_self_repetition_score(questions: List[str], n: int = 4) -> float:
    """
    Self-Repetition Score (SRS)
    Fraction of questions containing at least one n-gram that appears in another question
    """
    # Collect all n-grams per question
    question_ngrams = []
    all_ngrams = Counter()
    
    for question in questions:
        tokens = question.lower().split()
        ngrams = set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
        question_ngrams.append(ngrams)
        all_ngrams.update(ngrams)
    
    # Count questions with repeated n-grams
    repeated_count = 0
    for ngrams in question_ngrams:
        has_repeat = any(all_ngrams[ng] > 1 for ng in ngrams)
        if has_repeat:
            repeated_count += 1
    
    return repeated_count / len(questions) if questions else 0.0

def compute_compression_ratio(text: str) -> float:
    """Compression ratio using gzip (lower = more diverse)"""
    import gzip
    compressed = gzip.compress(text.encode('utf-8'))
    return len(compressed) / len(text.encode('utf-8'))

def compute_diversity_metrics(questions: List[str]) -> Dict[str, float]:
    """Compute all diversity metrics"""
    return {
        'ndg_score': compute_ndg_score(questions),
        'self_repetition_4gram': compute_self_repetition_score(questions, n=4),
        'compression_ratio': compute_compression_ratio(' '.join(questions)),
        'unique_questions': len(set(questions)),
        'total_questions': len(questions),
        'uniqueness_ratio': len(set(questions)) / len(questions) if questions else 0.0
    }

def bootstrap_diversity_ci(questions: List[str], n_bootstrap: int = 50, sample_frac: float = 0.8) -> Dict[str, tuple]:
    """
    Bootstrap confidence intervals for diversity metrics
    Uses sampling without replacement as per DataMorgana
    """
    sample_size = int(len(questions) * sample_frac)
    
    ndg_scores = []
    sr_scores = []
    cr_scores = []
    
    for _ in range(n_bootstrap):
        sample = random.sample(questions, sample_size)
        ndg_scores.append(compute_ndg_score(sample))
        sr_scores.append(compute_self_repetition_score(sample))
        cr_scores.append(compute_compression_ratio(' '.join(sample)))
    
    def ci(scores):
        mean = np.mean(scores)
        std = np.std(scores)
        return (mean, mean - 1.96*std, mean + 1.96*std)
    
    return {
        'ndg': ci(ndg_scores),
        'self_repetition': ci(sr_scores),
        'compression_ratio': ci(cr_scores)
    }



# Custom JSON serializer
def serialize_for_json(obj):
    """Handle numpy types and other non-JSON-serializable objects"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, tuple):
        return str(obj)
    elif isinstance(obj, dict):
        return {str(k): v for k, v in obj.items()}
    return str(obj)







# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def compute_generation_metrics(results: List[Dict[str, Any]], model) -> Dict[str, float]:
    """Compute generation metrics"""
    rouge1_scores = []
    bleu_scores = []
    cosine_scores = []
    refusal_count = 0
    
    for item in results:
        reference = item.get('reference_answer', item.get('answer', ''))
        generated = item.get('generated_answer', '')
        
        if not reference or not generated:
            continue
        
        rouge = compute_rouge(reference, generated)
        rouge1_scores.append(rouge['rouge1'])
        
        bleu = compute_bleu(reference, generated)
        bleu_scores.append(bleu)
        
        cos_sim = compute_cosine_similarity(reference, generated, model)
        cosine_scores.append(cos_sim)
        
        if is_refusal(generated):
            refusal_count += 1
    
    return {
        'rouge1': np.mean(rouge1_scores) if rouge1_scores else 0.0,
        'bleu': np.mean(bleu_scores) if bleu_scores else 0.0,
        'cosine_sim': np.mean(cosine_scores) if cosine_scores else 0.0,
        'cosine_sim_std': np.std(cosine_scores) if cosine_scores else 0.0,
        'refusal_rate': (refusal_count / len(results) * 100) if results else 0.0,
        'n_samples': len(results)
    }

def compute_retrieval_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute retrieval metrics"""
    map_scores = []
    ndcg_scores = []
    recall_at_10 = []
    
    for item in results:
        gold_docs = set(item.get('document_ids', []))
        if not gold_docs:
            continue
        
        retrieved = item.get('retrieved_document_ids', [])[:10]
        
        if not retrieved:
            map_scores.append(0.0)
            ndcg_scores.append(0.0)
            recall_at_10.append(0.0)
            continue
        
        # MAP
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

# ============================================================================
# RQ1: GRANULARITY ANALYSIS
# ============================================================================

def extract_category_for_categorization(item, categorization_name):
    """Extract specific category name for a categorization"""
    for qcat in item.get('question_categories', []):
        if qcat.get('categorization_name') == categorization_name:
            return qcat.get('category_name', 'unknown')
    return 'unknown'

def analyze_rq1(results_file: str, model):
    """RQ1: Granularity analysis"""
    print("\n" + "="*80)
    print("RQ1: GRANULARITY ANALYSIS")
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
        print(f"\n{cat.upper()}:")
        all_results[cat] = {}
        
        for gran in ['coarse', 'medium', 'fine']:
            items = grouped[cat][gran]
            if not items:
                continue
            
            print(f"\n  {gran} (n={len(items)}):")
            
            # Retrieval & generation
            retrieval = compute_retrieval_metrics(items)
            generation = compute_generation_metrics(items, model)
            
            print(f"    MAP: {retrieval['map']:.3f}")
            print(f"    CS: {generation['cosine_sim']:.3f}")
            
            # Discriminative power
            by_category = defaultdict(list)
            for item in items:
                cat_name = extract_category_for_categorization(item, cat)
                by_category[cat_name].append(item)
            
            category_means = []
            for cat_name, cat_items in by_category.items():
                gen_metrics = compute_generation_metrics(cat_items, model)
                category_means.append(gen_metrics['cosine_sim'])
            
            disc_pow = np.std(category_means) if len(category_means) > 1 else 0.0
            print(f"    DiscPow: {disc_pow:.3f}")
            
            all_results[cat][gran] = {
                'retrieval': retrieval,
                'generation': generation,
                'discriminative_power': disc_pow,
                'category_means': category_means,
                'n_categories': len(by_category)
            }
    
    return all_results

# ============================================================================
# RQ1 FINE-GRAINED BREAKDOWN
# ============================================================================

def analyze_rq1_fine_detailed(results_file: str, model):
    """Detailed per-category analysis for fine granularity"""
    print("\n" + "="*80)
    print("RQ1 FINE-GRAINED: Per-Category Performance")
    print("="*80)
    
    results = load_jsonl(results_file)
    
    fine_results = {}
    
    for cat in ['question_complexity', 'question_answertype', 'question_linguisticvariation']:
        print(f"\n{cat.upper()}:")
        
        # Get fine-grained items
        items = [r for r in results if r.get('categorization') == cat and r.get('granularity') == 'fine']
        
        # Group by specific category
        by_category = defaultdict(list)
        for item in items:
            cat_name = extract_category_for_categorization(item, cat)
            by_category[cat_name].append(item)
        
        category_results = {}
        for cat_name, cat_items in sorted(by_category.items()):
            retrieval = compute_retrieval_metrics(cat_items)
            generation = compute_generation_metrics(cat_items, model)
            
            print(f"  {cat_name}: n={len(cat_items)}, MAP={retrieval['map']:.3f}, CS={generation['cosine_sim']:.3f}")
            
            category_results[cat_name] = {
                'n': len(cat_items),
                'map': retrieval['map'],
                'cosine_sim': generation['cosine_sim']
            }
        
        fine_results[cat] = category_results
    
    return fine_results

# ============================================================================
# RQ2: COMPLEMENTARITY ANALYSIS
# ============================================================================

def analyze_rq2(results_file: str, model):
    """RQ2: Which dimensions are most discriminative?"""
    print("\n" + "="*80)
    print("RQ2: COMPLEMENTARITY ANALYSIS")
    print("="*80)
    
    results = load_jsonl(results_file)
    print(f"Loaded {len(results)} results")
    
    grouped = defaultdict(list)
    for item in results:
        cat_set = item.get('categorization_set', '')
        grouped[cat_set].append(item)
    
    all_results = {}
    
    set_names = {
        'complexity': 'Complexity',
        'answertype': 'Answer Type',
        'vocabulary': 'Vocabulary',
        'phrasing': 'Phrasing',
        'expertise': 'User Expertise'
    }
    
    for cat_set, display_name in set_names.items():
        items = grouped[cat_set]
        if not items:
            continue
        
        print(f"\n{display_name} (n={len(items)}):")
        
        retrieval = compute_retrieval_metrics(items)
        generation = compute_generation_metrics(items, model)
        
        # Performance range
        by_category = defaultdict(list)
        for item in items:
            for qcat in item.get('question_categories', []):
                by_category[qcat['category_name']].append(item)
        
        category_means = []
        for cat_items in by_category.values():
            gen_metrics = compute_generation_metrics(cat_items, model)
            category_means.append(gen_metrics['cosine_sim'])
        
        perf_range = max(category_means) - min(category_means) if category_means else 0.0
        
        print(f"  MAP: {retrieval['map']:.3f}")
        print(f"  CS: {generation['cosine_sim']:.3f}")
        print(f"  Range: {perf_range:.3f}")
        
        all_results[cat_set] = {
            'retrieval': retrieval,
            'generation': generation,
            'performance_range': perf_range
        }
    
    return all_results

# ============================================================================
# RQ3: INTERACTION ANALYSIS
# ============================================================================

def analyze_rq3(results_file: str, model):
    """RQ3: Vocabulary × Complexity interaction"""
    print("\n" + "="*80)
    print("RQ3: INTERACTION ANALYSIS")
    print("="*80)
    
    results = load_jsonl(results_file)
    print(f"Loaded {len(results)} results")
    
    # Group by 2×2 factorial
    factorial = {}
    
    for item in results:
        vocab_cat = None
        complexity_cat = None
        
        for qcat in item.get('question_categories', []):
            cat_name = qcat['categorization_name']
            category = qcat['category_name']
            
            if 'linguistic' in cat_name:
                vocab_cat = category
            elif 'complexity' in cat_name:
                complexity_cat = category
        
        if vocab_cat and complexity_cat:
            # Normalize category names to expected values
            # Map variations to standard names
            vocab_normalized = vocab_cat
            if 'similar' in vocab_cat.lower():
                vocab_normalized = 'similar_to_document'
            elif 'distant' in vocab_cat.lower():
                vocab_normalized = 'distant_from_document'
            
            comp_normalized = complexity_cat
            if complexity_cat.lower() in ['simple', 'single_fact', 'single-fact']:
                comp_normalized = 'simple'
            elif complexity_cat.lower() in ['complex', 'multi_fact', 'cross_section', 'comparative']:
                comp_normalized = 'complex'
            
            # Create short key for JSON serialization
            vocab_key = 's' if 'similar' in vocab_normalized else 'd'
            comp_key = 's' if comp_normalized == 'simple' else 'c'
            key = f"{vocab_key}|{comp_key}"
            
            if key not in factorial:
                factorial[key] = []
            factorial[key].append(item)
    
    # Debug output
    print("\n  Keys found:")
    for key in sorted(factorial.keys()):
        print(f"    {key}: n={len(factorial[key])}")
    
    all_results = {}
    
    # Expected keys: s|s, s|c, d|s, d|c
    key_labels = {
        's|s': ('similar_to_document', 'simple'),
        's|c': ('similar_to_document', 'complex'),
        'd|s': ('distant_from_document', 'simple'),
        'd|c': ('distant_from_document', 'complex')
    }
    
    for key in sorted(factorial.keys()):
        items = factorial[key]
        vocab_label, comp_label = key_labels.get(key, key.split('|'))
        
        retrieval = compute_retrieval_metrics(items)
        generation = compute_generation_metrics(items, model)
        
        print(f"\n  {vocab_label} × {comp_label} (n={len(items)}):")
        print(f"    MAP: {retrieval['map']:.3f}")
        print(f"    CS: {generation['cosine_sim']:.3f}")
        
        all_results[key] = {
            'n': len(items),
            'retrieval': retrieval,
            'generation': generation,
            'vocab_label': vocab_label,
            'comp_label': comp_label
        }
    
    # Compute main effects
    print("\n  Main Effects:")
    
    similar_items = [item for key, items in factorial.items() if key.startswith('s') for item in items]
    distant_items = [item for key, items in factorial.items() if key.startswith('d') for item in items]
    
    if similar_items and distant_items:
        similar_gen = compute_generation_metrics(similar_items, model)
        distant_gen = compute_generation_metrics(distant_items, model)
        
        print(f"    Similar vocab: CS={similar_gen['cosine_sim']:.3f}")
        print(f"    Distant vocab: CS={distant_gen['cosine_sim']:.3f}")
        print(f"    Gap: {similar_gen['cosine_sim'] - distant_gen['cosine_sim']:.3f}")
    
    return all_results

# ============================================================================
# BASELINE COMPARISON
# ============================================================================

def baseline_comparison(results_file: str, model):
    """Compare structured vs random sampling"""
    print("\n" + "="*80)
    print("BASELINE COMPARISON")
    print("="*80)
    
    results = load_jsonl(results_file)
    structured_questions = [item.get('question', '') for item in results if item.get('question')]
    
    print(f"\nStructured: n={len(structured_questions)}")
    
    # Random baseline
    random_sample = random.sample(results, len(results))
    random_questions = [item.get('question', '') for item in random_sample]
    
    # Diversity metrics
    print("\nDiversity Metrics:")
    
    structured_div = compute_diversity_metrics(structured_questions)
    random_div = compute_diversity_metrics(random_questions)
    
    print(f"\n  NDG Score:")
    print(f"    Structured: {structured_div['ndg_score']:.3f}")
    print(f"    Random: {random_div['ndg_score']:.3f}")
    print(f"    Gain: {(structured_div['ndg_score'] - random_div['ndg_score']) / random_div['ndg_score'] * 100:+.1f}%")
    
    print(f"\n  Self-Repetition:")
    print(f"    Structured: {structured_div['self_repetition_4gram']:.3f}")
    print(f"    Random: {random_div['self_repetition_4gram']:.3f}")
    
    # Semantic coverage
    print("\nSemantic Coverage:")
    
    structured_emb = model.encode(structured_questions)
    random_emb = model.encode(random_questions)
    
    # Mean pairwise distance
    structured_cos = cosine_similarity(structured_emb)
    random_cos = cosine_similarity(random_emb)
    
    np.fill_diagonal(structured_cos, np.nan)
    np.fill_diagonal(random_cos, np.nan)
    
    structured_dist = np.nanmean(1 - structured_cos)
    random_dist = np.nanmean(1 - random_cos)
    
    print(f"  Mean pairwise distance:")
    print(f"    Structured: {structured_dist:.3f}")
    print(f"    Random: {random_dist:.3f}")
    print(f"    Gain: {(structured_dist - random_dist) / random_dist * 100:+.1f}%")
    
    # Coverage radius
    structured_centroid = np.mean(structured_emb, axis=0)
    random_centroid = np.mean(random_emb, axis=0)
    
    structured_radius = np.mean([1 - cosine_similarity([emb], [structured_centroid])[0][0] for emb in structured_emb])
    random_radius = np.mean([1 - cosine_similarity([emb], [random_centroid])[0][0] for emb in random_emb])
    
    print(f"  Coverage radius:")
    print(f"    Structured: {structured_radius:.3f}")
    print(f"    Random: {random_radius:.3f}")
    print(f"    Gain: {(structured_radius - random_radius) / random_radius * 100:+.1f}%")
    
    # Performance spread
    print("\nPerformance Spread:")
    
    structured_cosines = []
    random_cosines = []
    
    for item in results:
        ref = item.get('reference_answer', item.get('answer', ''))
        gen = item.get('generated_answer', '')
        if ref and gen:
            structured_cosines.append(compute_cosine_similarity(ref, gen, model))
    
    for item in random_sample:
        ref = item.get('reference_answer', item.get('answer', ''))
        gen = item.get('generated_answer', '')
        if ref and gen:
            random_cosines.append(compute_cosine_similarity(ref, gen, model))
    
    print(f"  CS std:")
    print(f"    Structured: {np.std(structured_cosines):.3f}")
    print(f"    Random: {np.std(random_cosines):.3f}")
    print(f"    Gain: {(np.std(structured_cosines) - np.std(random_cosines)) / np.std(random_cosines) * 100:+.1f}%")
    
    print(f"  CS range:")
    structured_range = np.max(structured_cosines) - np.min(structured_cosines)
    random_range = np.max(random_cosines) - np.min(random_cosines)
    print(f"    Structured: {structured_range:.3f}")
    print(f"    Random: {random_range:.3f}")
    print(f"    Gain: {(structured_range - random_range) / random_range * 100:+.1f}%")
    
    return {
        'structured': {
            'diversity': structured_div,
            'semantic_distance': structured_dist,
            'coverage_radius': structured_radius,
            'perf_std': np.std(structured_cosines),
            'perf_range': structured_range
        },
        'random': {
            'diversity': random_div,
            'semantic_distance': random_dist,
            'coverage_radius': random_radius,
            'perf_std': np.std(random_cosines),
            'perf_range': random_range
        }
    }

# ============================================================================
# MUTUAL INFORMATION ANALYSIS
# ============================================================================

def mutual_information_analysis(results_file: str, model):
    """Compute MI between categories and performance"""
    print("\n" + "="*80)
    print("MUTUAL INFORMATION ANALYSIS")
    print("="*80)
    
    results = load_jsonl(results_file)
    
    data = []
    for item in results:
        ref = item.get('reference_answer', item.get('answer', ''))
        gen = item.get('generated_answer', '')
        
        if not ref or not gen:
            continue
        
        cos_sim = compute_cosine_similarity(ref, gen, model)
        
        data.append({
            'cosine_sim': cos_sim,
            'categorization': item.get('categorization', ''),
            'granularity': item.get('granularity', ''),
            'category': extract_category_for_categorization(item, item.get('categorization', ''))
        })
    
    df = pd.DataFrame(data)
    df['perf_bin'] = pd.cut(df['cosine_sim'], bins=5, labels=False)
    
    mi_results = {}
    
    for cat in ['question_complexity', 'question_answertype', 'question_linguisticvariation']:
        print(f"\n{cat.upper()}:")
        
        cat_df = df[df['categorization'] == cat]
        
        for gran in ['coarse', 'medium', 'fine']:
            gran_df = cat_df[cat_df['granularity'] == gran]
            
            if len(gran_df) < 10:
                continue
            
            mi = mutual_info_score(gran_df['category'], gran_df['perf_bin'])
            cat_entropy = entropy(gran_df['category'].value_counts(normalize=True))
            normalized_mi = mi / cat_entropy if cat_entropy > 0 else 0
            
            print(f"  {gran}: MI={mi:.4f}, normalized={normalized_mi:.4f}")
            
            mi_results[(cat, gran)] = {
                'mi': mi,
                'normalized_mi': normalized_mi,
                'n': len(gran_df)
            }
    
    return mi_results


# ============================================================================
# CALIBRATION ANALYSIS (Tree Structure Validation)
# ============================================================================

def compute_calibration_metrics(results_file: str, model):
    """
    Compute calibration scores to validate hierarchical structure.
    Measures vertical consistency (child-parent alignment) and horizontal 
    discrimination (sibling variance).
    """
    print("\n" + "="*80)
    print("CALIBRATION ANALYSIS: Hierarchical Structure Validation")
    print("="*80)
    
    results = load_jsonl(results_file)
    
    # Define hierarchical relationships
    hierarchy = {
        'question_complexity': {
            'coarse_to_medium': {
                'simple': ['single_fact', 'multi_fact_local'],
                'complex': ['cross_section_synthesis', 'comparative_analysis']
            },
            'medium_to_fine': {
                'single_fact': ['extractive_span', 'entity_extraction'],
                'multi_fact_local': ['multi_span_aggregation', 'paraphrasing_required'],
                'cross_section_synthesis': ['single_hop_inference', 'multi_hop_reasoning'],
                'comparative_analysis': ['comparative_synthesis', 'comparative_synthesis_concepts']
            }
        },
        'question_answertype': {
            'coarse_to_medium': {
                'extractive': ['short_span_extraction', 'list_or_enumeration'],
                'abstractive': ['summary_or_explanation', 'synthesis_or_analysis']
            },
            'medium_to_fine': {
                'short_span_extraction': ['entity_extraction', 'phrase_extraction'],
                'list_or_enumeration': ['unordered_list', 'ordered_sequence'],
                'summary_or_explanation': ['condensed_summary', 'sentence_extraction'],
                'synthesis_or_analysis': ['analytical_synthesis', 'explanatory_synthesis']
            }
        },
        'question_linguisticvariation': {
            'coarse_to_medium': {
                'similar_to_document': ['exact_terminology', 'partial_overlap'],
                'distant_from_document': ['synonym_substitution', 'conceptual_rephrase']
            },
            'medium_to_fine': {
                'exact_terminology': ['verbatim_terminology', 'high_lexical_overlap'],
                'partial_overlap': ['moderate_lexical_overlap', 'moderate_low_lexical_overlap'],
                'synonym_substitution': ['low_lexical_overlap', 'synonym_based_rephrase'],
                'conceptual_rephrase': ['domain_shift_terminology', 'abstraction_level_shift']
            }
        }
    }
    
    # Group results by category
    by_category = defaultdict(lambda: defaultdict(list))
    
    for item in results:
        cat_name = item.get('categorization', '')
        gran = item.get('granularity', '')
        
        for qcat in item.get('question_categories', []):
            if qcat['categorization_name'] == cat_name:
                category = qcat['category_name']
                by_category[cat_name][(gran, category)].append(item)
    
    # Compute performance for each category
    category_performance = {}
    for cat_name, cats in by_category.items():
        for (gran, category), items in cats.items():
            gen_metrics = compute_generation_metrics(items, model)
            ret_metrics = compute_retrieval_metrics(items)
            category_performance[(cat_name, gran, category)] = {
                'cosine_sim': gen_metrics['cosine_sim'],
                'map': ret_metrics['map'],
                'n': len(items)
            }
    
    # Compute calibration scores
    calibration_results = {}
    
    for cat_name, structure in hierarchy.items():
        print(f"\n{cat_name.upper()}:")
        calibration_results[cat_name] = {}
        
        # Medium→Fine calibration
        for medium_cat, fine_children in structure['medium_to_fine'].items():
            medium_perf = category_performance.get((cat_name, 'medium', medium_cat), {}).get('cosine_sim')
            
            if medium_perf is None:
                continue
            
            # Get fine children performance
            fine_perfs = []
            for fine_cat in fine_children:
                fine_perf = category_performance.get((cat_name, 'fine', fine_cat), {}).get('cosine_sim')
                if fine_perf is not None:
                    fine_perfs.append(fine_perf)
            
            if not fine_perfs:
                continue
            
            # Vertical consistency: deviation from parent
            mean_child_perf = np.mean(fine_perfs)
            vertical_deviation = abs(mean_child_perf - medium_perf)
            
            # Horizontal discrimination: variance among siblings
            sibling_variance = np.std(fine_perfs) if len(fine_perfs) > 1 else 0.0
            
            # Coherence ratio: high = good split (siblings differ but align with parent)
            # Analogous to silhouette coefficient in clustering
            coherence_ratio = sibling_variance / (vertical_deviation + 0.001)  # Avoid div by 0
            
            print(f"\n  {medium_cat} → {fine_children}:")
            print(f"    Parent (medium): CS={medium_perf:.3f}")
            print(f"    Children mean: CS={mean_child_perf:.3f}")
            print(f"    Vertical deviation: {vertical_deviation:.3f}")
            print(f"    Sibling variance: {sibling_variance:.3f}")
            print(f"    Coherence ratio: {coherence_ratio:.2f}")
            
            calibration_results[cat_name][medium_cat] = {
                'parent_perf': medium_perf,
                'children_mean': mean_child_perf,
                'children_perfs': fine_perfs,
                'vertical_deviation': vertical_deviation,
                'sibling_variance': sibling_variance,
                'coherence_ratio': coherence_ratio,
                'is_well_calibrated': vertical_deviation < 0.05 and sibling_variance > 0.02
            }
    
    # Summary statistics
    print("\n" + "="*80)
    print("CALIBRATION SUMMARY")
    print("="*80)
    
    all_vertical_devs = []
    all_sibling_vars = []
    all_coherence_ratios = []
    
    for cat_name, splits in calibration_results.items():
        for medium_cat, metrics in splits.items():
            all_vertical_devs.append(metrics['vertical_deviation'])
            all_sibling_vars.append(metrics['sibling_variance'])
            all_coherence_ratios.append(metrics['coherence_ratio'])
    
    print(f"\nMean vertical deviation: {np.mean(all_vertical_devs):.3f} (±{np.std(all_vertical_devs):.3f})")
    print(f"Mean sibling variance: {np.mean(all_sibling_vars):.3f} (±{np.std(all_sibling_vars):.3f})")
    print(f"Mean coherence ratio: {np.mean(all_coherence_ratios):.2f} (±{np.std(all_coherence_ratios):.2f})")
    
    # Interpretation
    print("\nInterpretation:")
    print("  - Vertical deviation: Lower = children align with parent (good hierarchy)")
    print("  - Sibling variance: Higher = fine split is discriminative (useful granularity)")
    print("  - Coherence ratio: Higher = split is both discriminative and well-aligned")
    print("    (Analogous to silhouette coefficient: ratio of between-cluster to within-cluster distance)")
    
    return calibration_results






# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("COMPLETE RAG ANALYSIS WITH DIVERSITY METRICS")
    print("="*80)
    
    # Load model
    print("\nLoading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Run all analyses
    rq1_results = analyze_rq1('./results/rq1_total_fixed.jsonl', model)
    rq1_fine = analyze_rq1_fine_detailed('./results/rq1_total_fixed.jsonl', model)
    rq2_results = analyze_rq2('./results/rq2_results_fixed.jsonl', model)
    rq3_results = analyze_rq3('./results/rq3_results_fixed.jsonl', model)
    baseline_results = baseline_comparison('./results/rq1_total_fixed.jsonl', model)
    mi_results = mutual_information_analysis('./results/rq1_total_fixed.jsonl', model)
    
    # Compute diversity metrics with bootstrap CI
    print("\n" + "="*80)
    print("DIVERSITY METRICS WITH BOOTSTRAP CI")
    print("="*80)
    
    all_questions = load_jsonl('./results/rq1_total_fixed.jsonl')
    questions = [item.get('question', '') for item in all_questions if item.get('question')]
    
    print(f"\nComputing bootstrap CI for {len(questions)} questions...")
    diversity_ci = bootstrap_diversity_ci(questions, n_bootstrap=50, sample_frac=0.8)
    
    print("\nDiversity Metrics (mean, 95% CI):")
    for metric, (mean, lower, upper) in diversity_ci.items():
        print(f"  {metric}: {mean:.3f} [{lower:.3f}, {upper:.3f}]")


    # Add to main() function before saving results:
    calibration_results = compute_calibration_metrics('./results/rq1_total_fixed.jsonl', model)



    # Save all results
    # Save all results - convert tuple keys to strings
    output = {
    'rq1': rq1_results,
    'rq1_fine': rq1_fine,
    'rq2': rq2_results,
    'rq3': {f"{k[0]}|{k[1]}": v for k, v in rq3_results.items()},  # Use | separator for tuple keys
    'baseline': baseline_results,
    'mutual_information': {f"{k[0]}|{k[1]}": v for k, v in mi_results.items()},  # Use | separator
    'diversity_ci': {k: {'mean': float(v[0]), 'lower': float(v[1]), 'upper': float(v[2])} 
                     for k, v in diversity_ci.items()},
    'calibration': calibration_results  # ADD THIS
    }
    
    with open('complete_analysis_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))

    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("✓ Saved to: complete_analysis_results.json")
    print("="*80)

if __name__ == "__main__":
    main()