# Extented Results

This document contains supplementary tables with complete experimental results referenced in the paper.

Abbreviations:

- Metrics:
MAP (Mean Average Precision); nDCG@10 (Normalized Discounted Cumulative Gain @ 10); ROUGE-1 (Recall-Oriented Understudy for Gisting Evaluation); BLEU (Bilingual Evaluation Understudy); CS (Cosine Similarity); NMI (Normalized Mutual Information); DiscPow (Discriminative Power, calculated here within a granularity as the standard deviation of cosine similarity scores across categories). For coherence ratio, see Eqn (1) of main text for definitions of ρ_coherence, δ_vert, and σ_sib.

- Dimensions:
LV (Linguistic Variation); AT (Answer Types); QC (Question Complexity). Phrasing = Phrasing for clarity.

---

## Table S1: Complete RQ1 Results (All Metrics)

### Complementarity Analysis across Four Dimensions: Full Retrieval and Generation Metrics

| Dimension | n | MAP | nDCG@10 | Recall@10 | ROUGE-1 | ROUGE-L | BLEU | CS | Refusal Rate |
|-----------|---|-----|---------|-----------|---------|---------|------|-----|--------------|
| LV | 400 | .369 | .367 | .435 | .337 | .247 | .088 | .695 | 19.5% |
| Phrasing | 400 | .503 | **.497** | **.485** | **.368** | .266 | .106 | .703 | 8.5% |
| AT | 400 | **.506** | **.497** | **.485** | **.368** | **.279** | **.117** | **.711** | 12.0% |
| QC | 400 | .478 | **.497** | **.485** | .366 | .276 | .115 | .704 | 15.5% |

**Key Finding:** LV shows highest CS range (.077) despite worst absolute performance, revealing inverse relationship between challenge level and discriminative power. **Bold** indicates best performance.

---

## Table S2: Complete RQ2 Results (All Granularity Levels x All Metrics)

### Question Complexity (QC)

| Granularity | n | MAP | nDCG@10 | Recall@10 | ROUGE-1 | ROUGE-L | BLEU | CS | NMI | DiscPow | Refusal Rate |
|-------------|---|-----|---------|-----------|---------|---------|------|-----|-----|---------|--------------|
| Coarse (2) | 333 | .518 | .497 | .485 | **.366** | **.276** | **.115** | **.712** | .008 | .007 | 17.5% |
| Medium (4) | 333 | **.564** | .564 | **.753** | .323 | .255 | .096 | .709 | .027 | .035 | 15.0% |
| Fine (8) | 446 | .562 | **.597** | **.753** | .324 | .256 | .097 | .708 | **.040** | **.053** | 14.5% |

### Answer Type (AT)

| Granularity | n | MAP | nDCG@10 | Recall@10 | ROUGE-1 | ROUGE-L | BLEU | CS | NMI | DiscPow | Refusal Rate |
|-------------|---|-----|---------|-----------|---------|---------|------|-----|-----|---------|--------------|
| Coarse (2) | 333 | .491 | .497 | .485 | **.368** | **.279** | **.117** | .692 | .028 | .024 | 17.0% |
| Medium (4) | 333 | **.514** | .514 | **.728** | .358 | .255 | .096 | **.699** | .026 | **.044** | 12.5% |
| Fine (8) | 382 | .504 | **.573** | **.728** | .349 | .249 | .089 | .684 | **.035** | .037 | 15.0% |

### Linguistic Variation (LV)

| Granularity | n | MAP | nDCG@10 | Recall@10 | ROUGE-1 | ROUGE-L | BLEU | CS | NMI | DiscPow | Refusal Rate |
|-------------|---|-----|---------|-----------|---------|---------|------|-----|-----|---------|--------------|
| Coarse (2) | 333 | .412 | **.367** | **.435** | **.337** | **.247** | **.088** | .672 | .049 | .039 | 19.5% |
| Medium (4) | 333 | .343 | .269 | .373 | .255 | .168 | .054 | .671 | **.054** | **.047** | 22.0% |
| Fine (8) | 446 | **.446** | **.367** | **.435** | .289 | .203 | .065 | **.699** | .020 | .031 | 18.0% |

**Key Finding:** QC shows monotonic DiscPow increase (.007→.035→.053); AT and LV peak at medium granularity. **Bold** indicates higher performance among the 3 granularity levels for a given dimension.

---

## Table S3: Per-Category Performance (All 24 Fine-Granularity Categories)

### Question Complexity - Fine Categories

| Category | Parent (Medium) | n | MAP | nDCG@10 | Recall@10 | ROUGE-1 | ROUGE-L | BLEU | CS | Refusal Rate |
|----------|----------------|---|-----|---------|-----------|---------|---------|------|-----|--------------|
| comparative_synthesis_concepts | comparative_analysis | 50 | .566 | .674 | .800 | .371 | .261 | .077 | **.766** | 2.0% |
| single_hop_inference | cross_section_synthesis | 59 | .495 | .596 | .755 | .395 | .305 | .111 | .755 | 1.7% |
| multi_hop_reasoning | cross_section_synthesis | 58 | .461 | .596 | .690 | .347 | .256 | .092 | .743 | 5.2% |
| comparative_synthesis | comparative_analysis | 65 | **.674** | **.710** | **.815** | **.438** | **.356** | **.173** | .753 | 4.6% |
| multi_span_aggregation | multi_fact_local | 51 | .553 | .605 | .706 | .423 | .310 | .138 | .705 | 7.8% |
| paraphrasing_required | multi_fact_local | 46 | .604 | .682 | .783 | .389 | .305 | .111 | .682 | 8.7% |
| entity_extraction | single_fact | 62 | .606 | .636 | .758 | .385 | .291 | .104 | .636 | 16.1% |
| extractive_span | single_fact | 55 | .523 | .619 | .691 | .388 | .309 | .111 | .619 | 20.0% |

### Answer Type - Fine Categories

| Category | Parent (Medium) | n | MAP | nDCG@10 | Recall@10 | ROUGE-1 | ROUGE-L | BLEU | CS | Refusal Rate |
|----------|----------------|---|-----|---------|-----------|---------|---------|------|-----|--------------|
| explanatory_synthesis | synthesis_or_analysis | 48 | .395 | .596 | .688 | .371 | .261 | .077 | **.738** | 6.3% |
| condensed_summary | summary_or_explanation | 44 | .349 | .573 | .682 | .340 | .256 | .068 | .725 | 6.8% |
| analytical_synthesis | synthesis_or_analysis | 47 | .458 | .622 | .750 | .365 | .266 | .082 | .703 | 8.5% |
| unordered_list | list_or_enumeration | 55 | .581 | **.662** | .782 | .412 | .303 | .131 | .698 | 9.1% |
| sentence_extraction | summary_or_explanation | 49 | .478 | .625 | .714 | .349 | .249 | .089 | .670 | 14.3% |
| entity_extraction | short_span_extraction | 55 | **.674** | .622 | **.800** | **.438** | **.356** | **.173** | .657 | 16.4% |
| phrase_extraction | short_span_extraction | 46 | .513 | .573 | .696 | .388 | .309 | .111 | .647 | 17.4% |
| ordered_sequence | list_or_enumeration | 38 | .547 | .547 | .737 | .412 | .303 | .131 | .625 | 18.4% |

### Linguistic Variation - Fine Categories

| Category | Parent (Medium) | n | MAP | nDCG@10 | Recall@10 | ROUGE-1 | ROUGE-L | BLEU | CS | Refusal Rate |
|----------|----------------|---|-----|---------|-----------|---------|---------|------|-----|--------------|
| abstraction_level_shift | conceptual_rephrase | 57 | .451 | .596 | .667 | .340 | .256 | .068 | **.729** | 10.5% |
| domain_shift_terminology | conceptual_rephrase | 60 | .333 | .573 | .617 | .226 | .167 | .015 | .722 | 11.7% |
| verbatim_terminology | exact_terminology | 44 | **.699** | **.753** | **.864** | **.453** | **.351** | **.150** | .717 | 12.8% |
| moderate_lexical_overlap | partial_overlap | 51 | .521 | .622 | .745 | .389 | .305 | .111 | .710 | 13.7% |
| moderate_low_lexical_overlap | partial_overlap | 53 | .549 | .625 | .755 | .385 | .291 | .104 | .707 | 15.1% |
| high_lexical_overlap | exact_terminology | 61 | .590 | .662 | .774 | .423 | .310 | .138 | .705 | 16.4% |
| synonym_based_rephrase | synonym_substitution | 59 | .293 | .547 | .594 | .347 | .256 | .092 | .683 | 18.6% |
| low_lexical_overlap | synonym_substitution | 61 | .225 | .367 | .516 | .255 | .168 | .054 | .624 | 25.8% |

**Key Finding:** Substantial 24% performance gap between best (comparative_synthesis_concepts: .766) and worst (extractive_span: .619) fine QC categories. Inverted hierarchy observed where some "complex" children outperform "simple" children. **Note**: categories with lower refusal rates tend to show higher CS, but this does not necessarily mean the answers are higher quality -- answer generation quality should be interpreted using CS, BLEU, ROUGE, and any other available metrics.

---

## Table S4: Complete RQ3 Interaction Results

### Full 2×2 Factorial: LV × QC with All Metrics

| LV | QC | n | MAP | nDCG@10 | Recall@10 | ROUGE-1 | ROUGE-L | BLEU | CS | Refusal Rate | CS Gap |
|----|----|----|-----|---------|-----------|---------|---------|------|-----|--------------|--------|
| Similar | Simple | 252 | .547 | .597 | .753 | .323 | .255 | .096 | .637 | 0.0% | — |
| Similar | Complex | 265 | .566 | .710 | .800 | .340 | .256 | .068 | .672 | 0.0% | +.035 |
| Distant | Simple | 265 | .237 | .269 | .373 | .255 | .168 | .054 | .578 | 37.4% | — |
| Distant | Complex | 218 | .164 | .260 | .358 | .226 | .167 | .015 | .578 | 37.6% | +.000 |

### Main Effects

| Effect | n | MAP | CS | CS Gap |
|--------|---|-----|-----|-----|
| Similar (pooled) | 517 | .557 | .655 | — |
| Distant (pooled) | 483 | .201 | .578 | -.077 |
| Simple (pooled) | 517 | .392 | .607 | — |
| Complex (pooled) | 483 | .365 | .625 | +.018 |

**Key Finding:** Additive effects—vocabulary dominates (Δ=.077, 13% relative) while complexity becomes irrelevant when retrieval fails (distant×complex gap = 0.0).

---

## Table S5: Hierarchical Calibration Scores (All Medium→Fine Splits)

### Question Complexity (QC) Calibration

| Parent (Medium) | Children (Fine) | Parent CS | Children Mean CS | δ_vert | σ_sib | ρ_coherence | Interpretation |
|----------------|----------------|-----------|-----------------|--------|-------|-------------|----------------|
| single_fact | extractive_span, entity_extraction | .651 | .627 | .023 | .009 | 0.37 | Low coherence: poor split |
| multi_fact_local | multi_span_aggregation, paraphrasing_required | .715 | .693 | .021 | .011 | 0.50 | Low coherence: moderate split |
| cross_section_synthesis | single_hop_inference, multi_hop_reasoning | .737 | .749 | .012 | .006 | 0.46 | Low coherence: siblings not aligned |
| comparative_analysis | comparative_synthesis, comparative_synthesis_concepts | .737 | .759 | .023 | .007 | 0.28 | Low coherence: poor split |

**QC Overall:** Mean ρ = 0.40 (±0.10) — Poor calibration across all splits despite highest DiscPow at fine.  See Eqn (1) of main text for definitions of ρ_coherence, δ_vert, and σ_sib.

### Answer Type (AT) Calibration

| Parent (Medium) | Children (Fine) | Parent CS | Children Mean CS | δ_vert | σ_sib | ρ_coherence | Interpretation |
|----------------|----------------|-----------|-----------------|--------|-------|-------------|----------------|
| short_span_extraction | entity_extraction, phrase_extraction | .622 | .652 | .029 | .005 | 0.17 | Very low coherence |
| list_or_enumeration | unordered_list, ordered_sequence | .728 | .662 | .067 | .036 | 0.53 | Moderate coherence |
| summary_or_explanation | condensed_summary, sentence_extraction | .705 | .697 | .007 | .027 | **3.31** | **Excellent coherence** |
| synthesis_or_analysis | analytical_synthesis, explanatory_synthesis | .730 | .721 | .009 | .017 | 1.76 | Good coherence |

**AT Overall:** Mean ρ = 1.44 (±1.30) — Best-calibrated dimension with one excellent split.

### Linguistic Variation (LV) Calibration

| Parent (Medium) | Children (Fine) | Parent CS | Children Mean CS | δ_vert | σ_sib | ρ_coherence | Interpretation |
|----------------|----------------|-----------|-----------------|--------|-------|-------------|----------------|
| exact_terminology | verbatim_terminology, high_lexical_overlap | .724 | .711 | .013 | .006 | 0.41 | Low coherence |
| partial_overlap | moderate_lexical_overlap, moderate_low_lexical_overlap | .718 | .708 | .009 | .002 | 0.16 | Very low coherence |
| synonym_substitution | low_lexical_overlap, synonym_based_rephrase | .618 | .654 | .036 | .029 | 0.79 | Moderate coherence |
| conceptual_rephrase | domain_shift_terminology, abstraction_level_shift | .639 | .726 | .086 | .004 | **0.04** | **Worst coherence** |

**LV Overall:** Mean ρ = 0.35 (±0.31) — Poor calibration; conceptual_rephrase split shows high vertical deviation.

---

## Table S6: Diversity Metrics (Bootstrap 95% Confidence Intervals)

### Lexical and Semantic Diversity of DataMorgana-Generated Synthetic Questions

| Metric | Value | 95% CI Lower | 95% CI Upper | Interpretation |
|--------|-------|--------------|--------------|----------------|
| N-Gram Diversity (NDG) | 0.659 | 0.658 | 0.661 | High lexical diversity |
| Self-Repetition (4-gram) | 0.542 | 0.530 | 0.554 | Moderate repetition |
| Compression Ratio | 0.361 | 0.360 | 0.362 | Low redundancy |

**Bootstrap Method:** 1000 resamples with replacement from all 5,872 questions.

**Key Finding:** High NDG (.659) indicates strong lexical diversity; moderate self-repetition (.542) suggests some phrase overlap between synthetic questions, but acceptable for synthetic generation.

---

## Table S7: Retrieval-vs-Generation Correlation Analysis

### Does Better Retrieval → Better Generation?

| LV Category | MAP | CS | Retrieval Rank | Generation Rank | Rank Correlation |
|-------------|-----|-----|----------------|-----------------|------------------|
| verbatim_terminology | .699 | .717 | 1 | 3 | Mismatch |
| high_lexical_overlap | .590 | .705 | 2 | 5 | Mismatch |
| abstraction_level_shift | .451 | .729 | 5 | **1** | **Strong Mismatch** |
| domain_shift_terminology | .333 | .722 | 6 | **2** | **Strong Mismatch** |
| synonym_based_rephrase | .293 | .683 | 7 | 6 | Aligned |
| low_lexical_overlap | .225 | .624 | 8 | 8 | Aligned |

**Spearman ρ:** 0.52 (p < .05) — Moderate positive correlation overall, but **strong disconnects** for distant-vocabulary categories. LV (Linguistic Variation).

**Key Finding:** Distant-vocabulary questions achieve high CS (.722-.729) despite poor retrieval (MAP=.225-.451), suggesting model compensates via world knowledge or hallucination.

---

## Table S8: Refusal Rate by Category Type

### Which Question Types Cause Most Refusals?

| Dimension | Category (Fine) | n | Refusal Rate | MAP | CS |
|-----------|----------------|---|--------------|-----|-----|
| LV | low_lexical_overlap | 61 | 25.8% | .225 | .624 |
| QC | extractive_span | 55 | 20.0% | .523 | .619 |
| AT | ordered_sequence | 38 | 18.4% | .547 | .625 |
| LV | synonym_based_rephrase | 59 | 18.6% | .293 | .683 |
| AT | phrase_extraction | 46 | 17.4% | .513 | .647 |

**Lowest Refusal Rates:**

| Dimension | Category (Fine) | n | Refusal Rate | MAP | CS |
|-----------|----------------|---|--------------|-----|-----|
| QC | single_hop_inference | 59 | 1.7% | .495 | .755 |
| QC | comparative_synthesis_concepts | 50 | 2.0% | .566 | .766 |
| AT | condensed_summary | 44 | 6.8% | .349 | .725 |

**Key Finding:** Refusal rate inversely correlates with retrieval quality (Pearson r = -0.68, p < .001). Low lexical overlap triggers most refusals (25.8%). **Note**: "refusal" to answer the question (Refusal Rate) is not inherenty a poor behavior for a model in question-answering if it not confident in its answer -- however all questions posed in our dataset are answerable using the FineWeb-10BT corpus (i.e., refusing to answer the question is not the most desirable outcome).

---

## Summary Statistics

### Overall Performance Across all 5,872 Questions (across RQ1, RQ2, and RQ3)

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| MAP | 0.464 | 0.152 | 0.164 | 0.699 |
| nDCG@10 | 0.512 | 0.147 | 0.260 | 0.753 |
| Recall@10 | 0.632 | 0.185 | 0.358 | 0.864 |
| ROUGE-1 | 0.356 | 0.089 | 0.226 | 0.453 |
| ROUGE-L | 0.268 | 0.074 | 0.167 | 0.356 |
| BLEU | 0.103 | 0.042 | 0.015 | 0.173 |
| CS | 0.697 | 0.048 | 0.619 | 0.766 |
| Refusal Rate | 14.2% | 7.8% | 0.0% | 37.6% |

---

**Data Availability:** Full per-question results (5,872 rows) available in `rag_system/generate/complete_analysis_results.json` in the repository.
