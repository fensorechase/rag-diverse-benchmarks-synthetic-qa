import pandas as pd
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def main():
    print("Generating Coherence vs. Agreement Plot...")

    # 1. Load Pre-Computed Coherence Ratios (System Metric)
    # derived from your 5000+ synthetic questions
    with open('complete_analysis_results.json', 'r') as f:
        results = json.load(f)
        calibration_data = results.get('calibration', {})

    # 2. Load Human Annotations (Human Signal)
    ann1 = pd.read_excel('./completed_human/COMPLETED_annotation_batch_annotator_1_HYBRID.xlsx')
    ann3 = pd.read_excel('./completed_human/COMPLETED_annotation_batch_annotator_3_HYBRID.xlsx')
    shared_ids = pd.read_csv('./annotate_files/shared_questions_list.csv')['qa_id'].astype(str).str.strip().tolist()

    # Clean and Merge Human Data
    ann1['qa_id'] = ann1['qa_id'].astype(str).str.strip()
    ann3['qa_id'] = ann3['qa_id'].astype(str).str.strip()
    
    paired = ann1[ann1['qa_id'].isin(shared_ids)].set_index('qa_id')[['category_name', 'category_validation']].join(
        ann3[ann3['qa_id'].isin(shared_ids)].set_index('qa_id')[['category_validation']], 
        lsuffix='_1', rsuffix='_2', how='inner'
    )

    # 3. Define Hierarchy for Matching (Parent <-> Child)
    # Maps ANY category label (Coarse, Medium, or Fine) to its 'Split'
    # Dictionary format: {Category_Label: (Dimension, Split_Name)}
    category_to_split = {}
    
    hierarchy = {
        'question_complexity': {
            'simple': ['single_fact', 'multi_fact_local'],
            'complex': ['cross_section_synthesis', 'comparative_analysis'],
            'single_fact': ['extractive_span', 'entity_extraction'], 
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

    # Flatten hierarchy for lookup
    for dim, splits in hierarchy.items():
        for parent, children in splits.items():
            # The Parent label validates the split
            category_to_split[parent] = (dim, parent)
            # The Child labels also validate the split
            for child in children:
                category_to_split[child] = (dim, parent)

    # 4. Match Questions to Coherence Ratios
    plot_data = []

    for qa_id, row in paired.iterrows():
        cat_gt = str(row['category_name']).strip()
        
        # Does this question belong to a known split?
        if cat_gt in category_to_split:
            dim, split_name = category_to_split[cat_gt]
            
            # Retrieve System Coherence for this split
            try:
                coherence = calibration_data[dim][split_name]['coherence_ratio']
            except KeyError:
                continue # Split might not exist in calibration data if filtered out
                
            # Retrieve Human Agreement
            is_agreed = 1 if row['category_validation_1'] == row['category_validation_2'] else 0
            
            plot_data.append({
                'Coherence Ratio': coherence,
                'Human Agreement': 'Agreed' if is_agreed else 'Disagreed',
                'Is_Agreed_Bin': is_agreed,
                'Split': split_name
            })

    df = pd.DataFrame(plot_data)
    print(f"Data points for correlation: {len(df)}")

    # 5. Calculate Statistic (Point-Biserial Correlation)
    if len(df) > 5:
        r, p = stats.pointbiserialr(df['Is_Agreed_Bin'], df['Coherence Ratio'])
        print(f"\nSTATISTIC: Point-Biserial r = {r:.3f}, p = {p:.3f}")
        
        with open('correlation_stat.txt', 'w') as f:
            f.write(f"r={r:.3f}, p={p:.3f}")

        # 6. Generate Plot
        plt.figure(figsize=(5, 4))
        sns.boxplot(x='Human Agreement', y='Coherence Ratio', data=df, palette="Set2", width=0.5)
        sns.stripplot(x='Human Agreement', y='Coherence Ratio', data=df, color=".25", size=4, jitter=True)
        
        plt.title(f"Coherence Ratio vs. Human Consensus\n(r={r:.2f}, p={p:.3f})")
        plt.ylabel("Split Coherence Ratio (System Metric)")
        plt.xlabel("Human Annotation Outcome")
        plt.tight_layout()
        plt.savefig('coherence_vs_agreement_plot.pdf')
        print("✓ Plot saved to: coherence_vs_agreement_plot.pdf")
    else:
        print("Not enough data points to plot.")

if __name__ == "__main__":
    main()