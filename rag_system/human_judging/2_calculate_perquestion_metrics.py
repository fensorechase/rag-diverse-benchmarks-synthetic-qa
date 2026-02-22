from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

df_all = pd.read_json('./annotate_files/combined_rq1_rq2_rq3_actual.jsonl', lines=True)

##################################################
# STEP 2: Calculate metrics for all questions (cosine similarity and MAP approximation)
##################################################

# Load embedding model for cosine similarity
print("Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def calculate_cosine_similarity(row):
    """Calculate cosine similarity between generated and reference answer"""
    if pd.isna(row.get('generated_answer')) or pd.isna(row.get('reference_answer')):
        return np.nan
    
    if row['generated_answer'] == '' or row['reference_answer'] == '':
        return np.nan
    
    gen_emb = model.encode([row['generated_answer']])
    ref_emb = model.encode([row['reference_answer']])
    
    return cosine_similarity(gen_emb, ref_emb)[0][0]

def calculate_map_approximation(row):
    """
    Approximate MAP based on whether ground-truth doc is in retrieved docs
    MAP = 1/(rank+1) where rank is position of first relevant doc
    """
    retrieved = row.get('retrieved_document_ids', [])
    ground_truth = row.get('document_ids', [])
    
    if not retrieved or not ground_truth:
        return np.nan
    
    # Check if ground truth doc appears in retrieved docs
    for i, doc_id in enumerate(retrieved):
        if doc_id in ground_truth:
            # MAP = 1/(rank+1)
            return 1.0 / (i + 1)
    
    return 0.0  # Not retrieved at all

print("Calculating cosine similarity for all questions...")
# Calculate in batches to show progress
batch_size = 1000
cs_scores = []

for i in range(0, len(df_all), batch_size):
    batch = df_all.iloc[i:i+batch_size]
    batch_scores = batch.apply(calculate_cosine_similarity, axis=1)
    cs_scores.extend(batch_scores)
    print(f"  Processed {min(i+batch_size, len(df_all))}/{len(df_all)} questions...")

df_all['cosine_similarity'] = cs_scores

print("\nCalculating MAP scores...")
df_all['map_score'] = df_all.apply(calculate_map_approximation, axis=1)

# Summary statistics
print("\n=== Metrics Summary ===")
print(f"Cosine Similarity:")
print(f"  Mean: {df_all['cosine_similarity'].mean():.3f}")
print(f"  Std:  {df_all['cosine_similarity'].std():.3f}")
print(f"  Min:  {df_all['cosine_similarity'].min():.3f}")
print(f"  Max:  {df_all['cosine_similarity'].max():.3f}")
print(f"  Missing: {df_all['cosine_similarity'].isna().sum()}")

print(f"\nMAP Score:")
print(f"  Mean: {df_all['map_score'].mean():.3f}")
print(f"  Std:  {df_all['map_score'].std():.3f}")
print(f"  Min:  {df_all['map_score'].min():.3f}")
print(f"  Max:  {df_all['map_score'].max():.3f}")
print(f"  Missing: {df_all['map_score'].isna().sum()}")

# Check by dimension
print("\n=== Metrics by Dimension ===")
for dim in ['QC', 'AT', 'LV', 'Phrasing']:
    dim_df = df_all[df_all['dimension'] == dim]
    print(f"\n{dim}:")
    print(f"  Mean CS:  {dim_df['cosine_similarity'].mean():.3f}")
    print(f"  Mean MAP: {dim_df['map_score'].mean():.3f}")

###############

# Check RQ1 only (matches Table 1)
print("\n=== RQ1 ONLY (Should Match Table 1) ===")
rq1_only = df_all[df_all['rq'] == 'RQ1']

for dim in ['QC', 'AT', 'LV', 'Phrasing']:
    dim_df = rq1_only[rq1_only['dimension'] == dim]
    print(f"\n{dim} (n={len(dim_df)}):")
    print(f"  Mean CS:  {dim_df['cosine_similarity'].mean():.3f}")
    print(f"  Mean MAP: {dim_df['map_score'].mean():.3f}")

# Save per-question metrics for reference
df_all.to_json('./annotate_files/combined_rq1_rq2_rq3_with_metrics.jsonl', orient='records', lines=True)




##################################################
# STEP 3: Stratified sample for human annotations
##################################################

def stratified_sample_for_annotation(df, n_total=150, seed=42):
    """
    Sample 150 examples stratified by:
    - RQ (RQ1: 50, RQ2: 75, RQ3: 25) - proportional to importance
    - Dimension (QC, AT, LV only for most - exclude Phrasing)
    - Granularity (within RQ2)
    - Performance tier (High/Low CS)
    """
    
    np.random.seed(seed)
    samples = []
    
    # --- RQ1: Sample 50 (coarse-level, diverse dimensions) ---
    print("\n=== Sampling RQ1 ===")
    rq1_df = df[df['rq'] == 'RQ1'].copy()
    
    # Include QC, AT, LV (exclude Phrasing - less critical for validation)
    rq1_df = rq1_df[rq1_df['dimension'].isin(['QC', 'AT', 'LV'])]
    
    for dim in ['QC', 'AT', 'LV']:
        dim_df = rq1_df[rq1_df['dimension'] == dim]
        
        if len(dim_df) == 0:
            continue
        
        # Split by performance (median CS)
        median_cs = dim_df['cosine_similarity'].median()
        high_perf = dim_df[dim_df['cosine_similarity'] >= median_cs]
        low_perf = dim_df[dim_df['cosine_similarity'] < median_cs]
        
        # Sample ~8 high + 8 low per dimension = ~48 total (16 per dim)
        n_high = 8
        n_low = 8
        
        if len(high_perf) >= n_high:
            sample_high = high_perf.sample(n=n_high, random_state=seed+len(samples))
            samples.append(sample_high)
            print(f"  {dim}: sampled {len(sample_high)} high-performing")
        else:
            samples.append(high_perf)
            print(f"  {dim}: sampled {len(high_perf)} high-performing (all available)")
        
        if len(low_perf) >= n_low:
            sample_low = low_perf.sample(n=n_low, random_state=seed+len(samples))
            samples.append(sample_low)
            print(f"  {dim}: sampled {len(sample_low)} low-performing")
        else:
            samples.append(low_perf)
            print(f"  {dim}: sampled {len(low_perf)} low-performing (all available)")
    
    rq1_count = sum(len(s) for s in samples)
    print(f"RQ1 total sampled: {rq1_count}")
    
    # --- RQ2: Sample 75 (hierarchical - MOST IMPORTANT) ---
    print("\n=== Sampling RQ2 ===")
    rq2_df = df[df['rq'] == 'RQ2'].copy()
    
    for dim in ['QC', 'AT', 'LV']:
        for gran in ['coarse', 'medium', 'fine']:
            gran_df = rq2_df[
                (rq2_df['dimension'] == dim) & 
                (rq2_df['granularity'] == gran)
            ]
            
            if len(gran_df) == 0:
                continue
            
            # Sample 4-5 per (dim × gran × performance tier)
            # Target: 3 dims × 3 gran × ~8 samples = ~72-75
            median_cs = gran_df['cosine_similarity'].median()
            high_perf = gran_df[gran_df['cosine_similarity'] >= median_cs]
            low_perf = gran_df[gran_df['cosine_similarity'] < median_cs]
            
            # Fine gets slightly more samples (most important)
            n_high = 5 if gran == 'fine' else 4
            n_low = 5 if gran == 'fine' else 4
            
            if len(high_perf) >= n_high:
                sample_high = high_perf.sample(n=n_high, random_state=seed+len(samples))
                samples.append(sample_high)
            else:
                samples.append(high_perf)
            
            if len(low_perf) >= n_low:
                sample_low = low_perf.sample(n=n_low, random_state=seed+len(samples))
                samples.append(sample_low)
            else:
                samples.append(low_perf)
            
            print(f"  {dim} {gran}: sampled {len(samples[-2]) + len(samples[-1])}")
    
    rq2_count = sum(len(s) for s in samples) - rq1_count
    print(f"RQ2 total sampled: {rq2_count}")
    
    # --- RQ3: Sample 25 (interaction - less critical) ---
    print("\n=== Sampling RQ3 ===")
    rq3_df = df[df['rq'] == 'RQ3'].copy()
    
    # RQ3 is 2×2 factorial: LV (similar/distant) × QC (simple/complex)
    # We need to identify which questions belong to which cell
    # Sample ~6 from each of 4 cells
    
    # Strategy: sample randomly from RQ3, ensuring some diversity
    # (We can't easily stratify by cells without knowing exact category mappings)
    if len(rq3_df) >= 25:
        rq3_sample = rq3_df.sample(n=25, random_state=seed)
        samples.append(rq3_sample)
        print(f"  RQ3: sampled 25 (random stratified)")
    else:
        samples.append(rq3_df)
        print(f"  RQ3: sampled {len(rq3_df)} (all available)")
    
    rq3_count = sum(len(s) for s in samples) - rq1_count - rq2_count
    print(f"RQ3 total sampled: {rq3_count}")
    
    # Combine all samples
    final_sample = pd.concat(samples, ignore_index=True)
    
    # Adjust to exactly n_total if needed
    if len(final_sample) > n_total:
        print(f"\nTrimming from {len(final_sample)} to {n_total}...")
        final_sample = final_sample.sample(n=n_total, random_state=seed)
    elif len(final_sample) < n_total:
        print(f"\nWarning: Only {len(final_sample)} samples available (target: {n_total})")
    
    print(f"\n=== Final Sample: {len(final_sample)} questions ===")
    
    return final_sample

# Generate annotation sample
annotation_df = stratified_sample_for_annotation(df_all, n_total=150, seed=42)

# Verify distribution
print("\n=== Sample Distribution ===")
print("\nBy RQ:")
print(annotation_df['rq'].value_counts().sort_index())

print("\nBy Dimension:")
print(annotation_df['dimension'].value_counts())

print("\nBy Granularity (RQ2 only):")
rq2_sample = annotation_df[annotation_df['rq'] == 'RQ2']
print(rq2_sample['granularity'].value_counts())

print("\nBy Dimension × Granularity (RQ2):")
print(rq2_sample.groupby(['dimension', 'granularity']).size())

print("\nPerformance distribution (CS quartiles):")
print(annotation_df['cosine_similarity'].describe())