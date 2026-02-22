import pandas as pd
import numpy as np

# Load files (with corrected labels)
rq2_actual = pd.read_json('../results/rq1_results_total.jsonl', lines=True)  # Mislabeled file
rq1_actual = pd.read_json('../results/rq2_results_total.jsonl', lines=True)  # Mislabeled file
rq3_actual = pd.read_json('../results/rq3_results_total.jsonl', lines=True)  # Correct

if 'qa_id' not in rq3_actual.columns and 'record_id' in rq3_actual.columns:
    rq3_actual['qa_id'] = rq3_actual['record_id']
    print("Fixed RQ3: renamed 'record_id' to 'qa_id'")

print(f"Total questions loaded: {len(rq1_actual) + len(rq2_actual) + len(rq3_actual)}")
print(f"RQ1 (actual): {len(rq1_actual)}")
print(f"RQ2 (actual): {len(rq2_actual)}")
print(f"RQ3 (actual): {len(rq3_actual)}")

# Relabel the 'rq' column correctly
rq1_actual['rq'] = 'RQ1'
rq2_actual['rq'] = 'RQ2'
rq3_actual['rq'] = 'RQ3'

# Combine
df_all = pd.concat([rq1_actual, rq2_actual, rq3_actual], ignore_index=True)

# Ensure qa_id exists for all rows
if df_all['qa_id'].isna().any():
    print(f"⚠ Warning: {df_all['qa_id'].isna().sum()} rows missing qa_id")
    # Generate qa_id for rows that don't have one
    import uuid
    df_all.loc[df_all['qa_id'].isna(), 'qa_id'] = [str(uuid.uuid4()) for _ in range(df_all['qa_id'].isna().sum())]
    print("Generated qa_id for missing rows")
    

# Extract dimension
def get_dimension(row):
    cats = row['question_categories']
    if not cats:
        return 'Unknown'
    
    cat_names = [c['categorization_name'] for c in cats]
    
    # RQ3: has both LV and QC
    if len(cats) >= 2:
        has_lv = any('linguistic' in c.lower() or 'variation' in c.lower() for c in cat_names)
        has_qc = any('complexity' in c.lower() for c in cat_names)
        if has_lv and has_qc:
            return 'LV'  # Label as LV for sampling purposes (or could use 'LV×QC')
    
    # Single categorization (RQ1/RQ2)
    cat_name = cats[0]['categorization_name']
    
    if 'complexity' in cat_name.lower():
        return 'QC'
    elif 'answertype' in cat_name.lower():
        return 'AT'
    elif 'linguistic' in cat_name.lower() or 'variation' in cat_name.lower():
        return 'LV'
    elif 'phrasing' in cat_name.lower():
        return 'Phrasing'
    elif 'factuality' in cat_name.lower():
        return 'Factuality_UserExpertise'
    else:
        return 'Unknown'

def get_category_name(row):
    cats = row['question_categories']
    if not cats:
        return 'Unknown'
    return cats[0]['category_name']

def get_user_expertise(row):
    user_cats = row.get('user_categories', [])
    if not user_cats:
        return 'Unknown'
    return user_cats[0]['category_name']

# Add derived columns
df_all['dimension'] = df_all.apply(get_dimension, axis=1)
df_all['category_name'] = df_all.apply(get_category_name, axis=1)
df_all['user_expertise'] = df_all.apply(get_user_expertise, axis=1)

# **EXCLUDE the dropped User Expertise dimension (400 questions)**
df_all = df_all[df_all['dimension'] != 'Factuality_UserExpertise']

print(f"\nAfter excluding User Expertise dimension: {len(df_all)} questions")
print("\nDimension distribution:")
print(df_all['dimension'].value_counts())
print("\nRQ distribution:")
print(df_all['rq'].value_counts())
print("\nGranularity distribution:")
print(df_all['granularity'].value_counts())

df_all.to_json('./annotate_files/combined_rq1_rq2_rq3_actual.jsonl', orient='records', lines=True)