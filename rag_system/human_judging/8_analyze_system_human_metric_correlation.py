import pandas as pd
import numpy as np
from scipy import stats
import json
from collections import defaultdict, Counter
from typing import List, Dict, Any
import re
from tqdm import tqdm
import random

# NLP libraries
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy

import sys
import os
# Adds the parallel directory 'generate' to the search path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from generate.complete_analysis import *

random.seed(42)
np.random.seed(42)




def compute_calibration_metrics(all_results: list, model):
    print("\n" + "="*80)
    print("CALIBRATION ANALYSIS: Hierarchical Structure (Coarse & Medium)")
    print("="*80)
    
    results = all_results

    # Full Hierarchy
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
    
    # Group results
    by_category = defaultdict(lambda: defaultdict(list))
    for item in results:
        cat_name = item.get('categorization', '')
        gran = item.get('granularity', '')
        for qcat in item.get('question_categories', []):
            if qcat['categorization_name'] == cat_name:
                category = qcat['category_name']
                by_category[cat_name][(gran, category)].append(item)
    
    # Compute performance
    category_performance = {}
    for cat_name, cats in by_category.items():
        for (gran, category), items in cats.items():
            gen_metrics = compute_generation_metrics(items, model)
            category_performance[(cat_name, gran, category)] = {
                'cosine_sim': gen_metrics['cosine_sim'],
                'n': len(items)
            }
    
    calibration_results = {}
    
    # Helper to calculate coherence for a single level
    def process_level(dim_name, structure, level_name, parent_gran, child_gran):
        results = {}
        for parent_cat, children in structure.items():
            # Get Parent Performance
            parent_perf = category_performance.get((dim_name, parent_gran, parent_cat), {}).get('cosine_sim')
            if parent_perf is None: continue
            
            # Get Children Performance
            child_perfs = []
            valid_children = []
            for child in children:
                p = category_performance.get((dim_name, child_gran, child), {}).get('cosine_sim')
                if p is not None:
                    child_perfs.append(p)
                    valid_children.append(child)
            
            if not child_perfs: continue
            
            # CALC METRICS
            mean_child_perf = np.mean(child_perfs)
            vertical_deviation = abs(mean_child_perf - parent_perf)
            sibling_variance = np.std(child_perfs) if len(child_perfs) > 1 else 0.0
            coherence_ratio = sibling_variance / (vertical_deviation + 0.001)
            
            results[parent_cat] = {
                'level': level_name,
                'children': valid_children,
                'coherence_ratio': coherence_ratio
            }
        return results

    # Main Loop
    for cat_name, structure in hierarchy.items():
        calibration_results[cat_name] = {}
        
        # 1. Coarse -> Medium (NEW)
        coarse_results = process_level(cat_name, structure['coarse_to_medium'], 'coarse_to_medium', 'coarse', 'medium')
        calibration_results[cat_name].update(coarse_results)
        
        # 2. Medium -> Fine (EXISTING)
        medium_results = process_level(cat_name, structure['medium_to_fine'], 'medium_to_fine', 'medium', 'fine')
        calibration_results[cat_name].update(medium_results)

    return calibration_results




def analyze_metric_correlation(calibration_results, ann1_path, ann3_path, shared_ids_path):
    print("\n" + "="*80)
    print("METRIC VALIDATION: Coherence Ratio vs. Human Signal (Coarse & Medium)")
    print("="*80)

    # 1. Load Data
    try:
        ann1 = pd.read_excel(ann1_path)
        ann3 = pd.read_excel(ann3_path)
        shared_ids = pd.read_csv(shared_ids_path)['qa_id'].astype(str).str.strip().tolist()
    except Exception as e:
        print(f"Error loading files: {e}")
        return {}

    # 2. Hierarchy Map (Parent -> Children) for BOTH levels
    # We construct this dynamically from the calibration results keys or hardcode it
    # Hardcoding ensures we catch everything even if calibration missed some data
    hierarchy = {
        'question_complexity': {
            'simple': ['single_fact', 'multi_fact_local'], # Coarse -> Medium
            'complex': ['cross_section_synthesis', 'comparative_analysis'],
            'single_fact': ['extractive_span', 'entity_extraction'], # Medium -> Fine
            'multi_fact_local': ['multi_span_aggregation', 'paraphrasing_required'],
            'cross_section_synthesis': ['single_hop_inference', 'multi_hop_reasoning'],
            'comparative_analysis': ['comparative_synthesis', 'comparative_synthesis_concepts']
        },
        'question_answertype': {
            'extractive': ['short_span_extraction', 'list_or_enumeration'],
            'abstractive': ['summary_or_explanation', 'synthesis_or_analysis'],
            'short_span_extraction': ['entity_extraction', 'phrase_extraction'],
            'list_or_enumeration': ['unordered_list', 'ordered_sequence'],
            'summary_or_explanation': ['condensed_summary', 'sentence_extraction'],
            'synthesis_or_analysis': ['analytical_synthesis', 'explanatory_synthesis']
        },
        'question_linguisticvariation': {
            'similar_to_document': ['exact_terminology', 'partial_overlap'],
            'distant_from_document': ['synonym_substitution', 'conceptual_rephrase'],
            'exact_terminology': ['verbatim_terminology', 'high_lexical_overlap'],
            'partial_overlap': ['moderate_lexical_overlap', 'moderate_low_lexical_overlap'],
            'synonym_substitution': ['low_lexical_overlap', 'synonym_based_rephrase'],
            'conceptual_rephrase': ['domain_shift_terminology', 'abstraction_level_shift']
        }
    }

    # 3. Prepare Paired Data
    ann1['qa_id'] = ann1['qa_id'].astype(str).str.strip()
    ann3['qa_id'] = ann3['qa_id'].astype(str).str.strip()
    
    paired = ann1[ann1['qa_id'].isin(shared_ids)].set_index('qa_id')[['category_name', 'category_validation']].join(
        ann3[ann3['qa_id'].isin(shared_ids)].set_index('qa_id')[['category_validation']], 
        lsuffix='_1', rsuffix='_2', how='inner'
    )
    
    print(f"DEBUG: Total Paired Questions: {len(paired)}")

    # 4. Question-Level Analysis
    question_level_data = []

    for dim_name, splits in calibration_results.items():
        # Short name
        if 'complexity' in dim_name: short_dim = 'QC'
        elif 'answertype' in dim_name: short_dim = 'AT'
        elif 'linguistic' in dim_name: short_dim = 'LV'
        else: short_dim = 'UNK'

        for parent_cat, metrics in splits.items():
            coherence = metrics['coherence_ratio']
            
            # VALIDATION LOGIC:
            # A split (Parent -> Children) is validated by questions that are:
            # 1. Labeled as the Parent (e.g., GT='simple' validates Simple->Single/Multi)
            # 2. Labeled as any of the Children (e.g., GT='single_fact' validates Simple->Single/Multi)
            
            children = hierarchy[dim_name].get(parent_cat, [])
            valid_gt_labels = [parent_cat] + children
            
            # Find matching questions
            mask = paired['category_name'].astype(str).str.strip().isin(valid_gt_labels)
            split_subset = paired[mask]
            
            if len(split_subset) == 0:
                continue

            for idx, row in split_subset.iterrows():
                is_agreed = 1 if row['category_validation_1'] == row['category_validation_2'] else 0
                
                question_level_data.append({
                    'Split_Name': parent_cat,
                    'Level': metrics['level'], # coarse_to_medium or medium_to_fine
                    'Coherence_Ratio': coherence,
                    'Is_Agreed': is_agreed,
                    'QA_ID': idx
                })

    # 5. Stats
    if not question_level_data:
        print("No matches found.")
        return {}

    df_q = pd.DataFrame(question_level_data)
    
    # Remove duplicates if a question validates multiple levels (rare but possible)
    # Actually, a 'single_fact' question validates BOTH 'simple' (its parent) and 'single_fact' (itself)
    # We should keep both because it effectively tests both hierarchies.
    
    print("\nMETRIC COMPARISON TABLE (Per Split):")
    stats_df = df_q.groupby('Split_Name').agg({
        'Coherence_Ratio': 'first',
        'Is_Agreed': 'mean',
        'QA_ID': 'count'
    }).rename(columns={'QA_ID': 'N_Samples'}).sort_values('Coherence_Ratio', ascending=False)
    print(stats_df)
    
    # Point-Biserial Correlation
    if len(df_q) > 4:
        r, p = stats.pointbiserialr(df_q['Is_Agreed'], df_q['Coherence_Ratio'])
        print(f"\nQUESTION-LEVEL CORRELATION (n={len(df_q)} interactions):")
        print(f"Point-Biserial r: {r:.3f}")
        print(f"P-value:          {p:.3f}")
        return {'correlation_r': r, 'correlation_p': p, 'data': stats_df.to_dict('index')}
    
    return {}

################################


#################################
def main():
    print("="*80)
    print("COMPLETE RAG ANALYSIS WITH DIVERSITY & CALIBRATION VALIDATION")
    print("="*80)
    
    # 1. Load Embedding Model
    print("\nLoading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # 2. Load ALL Data (RQ1 + RQ2 + RQ3)
    # This ensures we have the fine-grained examples from RQ2 for calibration
    print("\nLoading all datasets...")
    data_rq1 = load_jsonl('../results/rq1_results_total.jsonl')
    data_rq2 = load_jsonl('../results/rq2_results_total.jsonl')
    data_rq3 = load_jsonl('../results/rq3_results_total.jsonl')
    
    # Combine them into one big list for calibration
    all_data = data_rq1 + data_rq2 + data_rq3
    print(f"Total questions for calibration: {len(all_data)}")
    
    # 3. Compute Calibration (Using the COMBINED data)
    calibration_results = compute_calibration_metrics(all_data, model)
    
    # 4. Run Metric Correlation (Validating against human annotations)
    human_correlation = analyze_metric_correlation(
        calibration_results=calibration_results,
        ann1_path='./completed_human/COMPLETED_annotation_batch_annotator_1_HYBRID.xlsx',
        ann3_path='./completed_human/COMPLETED_annotation_batch_annotator_3_HYBRID.xlsx',
        shared_ids_path='./annotate_files/shared_questions_list.csv'
    )

    # Run all analyses
    rq1_results = analyze_rq1('../results/rq1_results_total.jsonl', model)
    #rq1_fine = analyze_rq1_fine_detailed('../results/rq1_results_total.jsonl', model)
    #rq2_results = analyze_rq2('../results/rq2_results_total.jsonl', model)
    #rq3_results = analyze_rq3('../results/rq3_results_total.jsonl', model)
    #baseline_results = baseline_comparison('../results/rq1_results_total.jsonl', model)
    #mi_results = mutual_information_analysis('../results/rq1_results_total.jsonl', model)
  
    # 5. Save Results
    output = {
        'rq1': rq1_results,
        #'rq1_fine': rq1_fine,
        #'rq2': rq2_results,
        #'rq3': {f"{k[0]}|{k[1]}": v for k, v in rq3_results.items()},
        'calibration': calibration_results,
        'human_metric_validation': human_correlation
    }
    
    with open('complete_analysis_results.json', 'w') as f:
            json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    
    print("\n" + "="*80)
    print("Analysis complete")
    print("Saved to: complete_analysis_results.json")
    print("="*80)

if __name__ == "__main__":
    main()