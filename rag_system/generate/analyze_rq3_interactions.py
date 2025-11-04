"""
analyze_rq3_interactions.py
Detailed analysis of vocabulary x complexity interaction effects
"""
import json
import numpy as np
import scipy.stats as stats
from compute_rag_metrics import (
    load_jsonl, compute_generation_metrics, compute_retrieval_metrics
)

def factorial_anova_rq3(results_file: str = './results/rq3_results.jsonl'):
    """Perform 2-way ANOVA on RQ3 factorial design"""
    print("\n" + "="*80)
    print("RQ3: FACTORIAL ANOVA ANALYSIS")
    print("="*80)
    
    results = load_jsonl(results_file)
    
    # Extract factors and dependent variable (cosine similarity)
    data = []
    for item in results:
        vocab_cat = None
        complexity_cat = None
        
        for qcat in item.get('question_categories', []):
            if qcat['categorization_name'] == 'question_linguisticvariation':
                vocab_cat = qcat['category_name']
            elif qcat['categorization_name'] == 'question_complexity':
                complexity_cat = qcat['category_name']
        
        if vocab_cat and complexity_cat:
            reference = item.get('reference_answer', item.get('answer', ''))
            generated = item.get('generated_answer', '')
            
            if reference and generated:
                from compute_rag_metrics import compute_cosine_similarity
                cos_sim = compute_cosine_similarity(reference, generated)
                data.append({
                    'vocab': vocab_cat,
                    'complexity': complexity_cat,
                    'cosine_sim': cos_sim
                })
    
    # Compute means by cell
    from collections import defaultdict
    cells = defaultdict(list)
    for d in data:
        key = (d['vocab'], d['complexity'])
        cells[key].append(d['cosine_sim'])
    
    print("\nCell Means (Cosine Similarity):")
    print(f"{'Vocabulary':<25} {'Complexity':<15} {'Mean':<10} {'Std':<10} {'N'}")
    print("-" * 70)
    
    cell_means = {}
    for (vocab, comp), values in sorted(cells.items()):
        mean_val = np.mean(values)
        std_val = np.std(values)
        cell_means[(vocab, comp)] = mean_val
        print(f"{vocab:<25} {comp:<15} {mean_val:.3f}      {std_val:.3f}      {len(values)}")
    
    # Main effects
    print("\nMain Effects:")
    
    # Vocabulary main effect
    similar_vals = [v for (vocab, _), vals in cells.items() if 'similar' in vocab for v in vals]
    distant_vals = [v for (vocab, _), vals in cells.items() if 'distant' in vocab for v in vals]
    print(f"  Similar vocabulary: {np.mean(similar_vals):.3f} (±{np.std(similar_vals):.3f})")
    print(f"  Distant vocabulary: {np.mean(distant_vals):.3f} (±{np.std(distant_vals):.3f})")
    print(f"  Difference: {np.mean(similar_vals) - np.mean(distant_vals):.3f}")
    t_stat, p_val = stats.ttest_ind(similar_vals, distant_vals)
    print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")
    
    # Complexity main effect
    simple_vals = [v for (_, comp), vals in cells.items() if comp == 'simple' for v in vals]
    complex_vals = [v for (_, comp), vals in cells.items() if comp == 'complex' for v in vals]
    print(f"\n  Simple questions: {np.mean(simple_vals):.3f} (±{np.std(simple_vals):.3f})")
    print(f"  Complex questions: {np.mean(complex_vals):.3f} (±{np.std(complex_vals):.3f})")
    print(f"  Difference: {np.mean(simple_vals) - np.mean(complex_vals):.3f}")
    t_stat, p_val = stats.ttest_ind(simple_vals, complex_vals)
    print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")
    
    # Interaction effect (visual)
    print("\nInteraction Pattern:")
    for vocab in ['similar_to_document', 'distant_from_document']:
        simple_mean = cell_means.get((vocab, 'simple'), 0)
        complex_mean = cell_means.get((vocab, 'complex'), 0)
        diff = simple_mean - complex_mean
        print(f"  {vocab}: simple={simple_mean:.3f}, complex={complex_mean:.3f}, diff={diff:.3f}")
    
    return cell_means

if __name__ == "__main__":
    factorial_anova_rq3()