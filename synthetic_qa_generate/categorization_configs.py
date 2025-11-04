# ============================================================================
# RQ1: GRANULARITY CATEGORIZATIONS
# ============================================================================

# ----------------------------------------------------------------------------
# Question Complexity - 3 Granularity Levels
# ----------------------------------------------------------------------------

question_complexity_coarse = {
    "categorization_name": "question_complexity",
    "categories": [
        {
            "name": "simple",
            "description": "A question that can be answered using information from a single, localized section of the document without requiring synthesis across multiple parts.",
            "probability": 0.5
        },
        {
            "name": "complex",
            "description": "A question that requires synthesizing or connecting information from multiple sections of the document or making inferences beyond directly stated facts.",
            "probability": 0.5
        }
    ]
}

question_complexity_medium = {
    "categorization_name": "question_complexity",
    "categories": [
        {
            "name": "single_fact",
            "description": "A question asking for one specific fact, date, name, or piece of information that appears explicitly in one location in the document.",
            "probability": 0.25
        },
        {
            "name": "multi_fact_local",
            "description": "A question requiring multiple facts or pieces of information that appear within the same paragraph or closely related section of the document.",
            "probability": 0.25
        },
        {
            "name": "cross_section_synthesis",
            "description": "A question requiring the reader to connect or synthesize information from multiple distinct sections of the document that are not immediately adjacent.",
            "probability": 0.25
        },
        {
            "name": "comparative_analysis",
            "description": "A question requiring comparison, contrast, or relational analysis between different entities, concepts, time periods, or perspectives discussed in the document.",
            "probability": 0.25
        }
    ]
}

question_complexity_fine = {
    "categorization_name": "question_complexity",
    "categories": [
        {
            "name": "extractive_span",
            "description": "A question whose answer is a single contiguous text span (phrase or sentence) that can be directly copied from one location in the document without any modification.",
            "probability": 0.167
        },
        {
            "name": "multi_span_aggregation",
            "description": "A question requiring aggregation of multiple text spans from the same section (e.g., listing several items, combining related facts from nearby sentences).",
            "probability": 0.167
        },
        {
            "name": "paraphrasing_required",
            "description": "A question whose answer requires paraphrasing or restating information from the document in different words, even though the information comes from one section.",
            "probability": 0.167
        },
        {
            "name": "single_hop_inference",
            "description": "A question requiring one logical inference step beyond what is explicitly stated, such as identifying an implication, consequence, or reason based on stated facts.",
            "probability": 0.167
        },
        {
            "name": "multi_hop_reasoning",
            "description": "A question requiring multiple connected reasoning steps, such as chaining together facts from different parts of the document to reach a conclusion.",
            "probability": 0.166
        },
        {
            "name": "comparative_synthesis",
            "description": "A question requiring structured comparison and synthesis, such as identifying similarities/differences, trade-offs, or relationships between multiple entities or concepts across the document.",
            "probability": 0.166
        }
    ]
}

# ----------------------------------------------------------------------------
# Answer Type - 3 Granularity Levels
# ----------------------------------------------------------------------------

question_answertype_coarse = {
    "categorization_name": "question_answertype",
    "categories": [
        {
            "name": "extractive",
            "description": "A question whose answer can be directly extracted as one or more contiguous text spans from the document without requiring rephrasing or synthesis.",
            "probability": 0.5
        },
        {
            "name": "abstractive",
            "description": "A question whose answer requires generating new text through paraphrasing, summarizing, or synthesizing information rather than extracting exact phrases from the document.",
            "probability": 0.5
        }
    ]
}

question_answertype_medium = {
    "categorization_name": "question_answertype",
    "categories": [
        {
            "name": "short_span_extraction",
            "description": "A question whose answer is a short phrase or single sentence that can be copied verbatim from the document (e.g., a name, date, or brief definition).",
            "probability": 0.25
        },
        {
            "name": "list_or_enumeration",
            "description": "A question whose answer is a list of items, steps, or elements that requires collecting and potentially organizing information from one or more parts of the document.",
            "probability": 0.25
        },
        {
            "name": "summary_or_explanation",
            "description": "A question whose answer requires summarizing a concept, process, or event described in the document using condensed or rephrased language.",
            "probability": 0.25
        },
        {
            "name": "synthesis_or_analysis",
            "description": "A question whose answer requires synthesizing multiple pieces of information or analyzing relationships, causes, effects, or implications discussed in the document.",
            "probability": 0.25
        }
    ]
}

question_answertype_fine = {
    "categorization_name": "question_answertype",
    "categories": [
        {
            "name": "entity_extraction",
            "description": "A question asking for a specific named entity (person, place, organization, date, number) that appears explicitly in the document.",
            "probability": 0.143
        },
        {
            "name": "phrase_extraction",
            "description": "A question whose answer is a short descriptive phrase or clause (not just an entity) that can be extracted verbatim from the document.",
            "probability": 0.143
        },
        {
            "name": "sentence_extraction",
            "description": "A question whose answer is best expressed as a complete sentence or two that can be extracted directly from the document with minimal modification.",
            "probability": 0.143
        },
        {
            "name": "unordered_list",
            "description": "A question whose answer is a collection of items, facts, or elements where the order does not matter (e.g., 'What are the main features of X?').",
            "probability": 0.143
        },
        {
            "name": "ordered_sequence",
            "description": "A question whose answer is a sequence of steps, events, or items where order matters (e.g., 'What are the steps in this process?').",
            "probability": 0.143
        },
        {
            "name": "condensed_summary",
            "description": "A question requiring a brief summary (2-4 sentences) that captures the main points about a topic discussed at length in the document.",
            "probability": 0.143
        },
        {
            "name": "analytical_synthesis",
            "description": "A question requiring analysis and synthesis beyond summarization, such as explaining relationships, comparing options, or describing implications using information from the document.",
            "probability": 0.142
        }
    ]
}

# ----------------------------------------------------------------------------
# Linguistic Variation - 3 Granularity Levels
# ----------------------------------------------------------------------------

question_linguisticvariation_coarse = {
    "categorization_name": "question_linguisticvariation",
    "categories": [
        {
            "name": "similar_to_document",
            "description": "A question phrased using the same terminology, phrases, and vocabulary that appear in the document.",
            "probability": 0.5
        },
        {
            "name": "distant_from_document",
            "description": "A question phrased using completely different terminology, synonyms, and expressions than those appearing in the document.",
            "probability": 0.5
        }
    ]
}

question_linguisticvariation_medium = {
    "categorization_name": "question_linguisticvariation",
    "categories": [
        {
            "name": "exact_terminology",
            "description": "A question using key terms and phrases that appear verbatim in the document, making lexical matching straightforward.",
            "probability": 0.25
        },
        {
            "name": "partial_overlap",
            "description": "A question where some key terms match the document while others use synonyms or related expressions, creating moderate lexical distance.",
            "probability": 0.25
        },
        {
            "name": "synonym_substitution",
            "description": "A question where most or all key terms are replaced with synonyms or semantically equivalent phrases not found in the document.",
            "probability": 0.25
        },
        {
            "name": "conceptual_rephrase",
            "description": "A question that describes the same concept or entity as the document but using entirely different vocabulary and conceptual framing (e.g., 'vehicle safety features' vs 'crash protection systems').",
            "probability": 0.25
        }
    ]
}

question_linguisticvariation_fine = {
    "categorization_name": "question_linguisticvariation",
    "categories": [
        {
            "name": "verbatim_terminology",
            "description": "A question using exact multi-word phrases and technical terms as they appear in the document, enabling direct string matching.",
            "probability": 0.167
        },
        {
            "name": "high_lexical_overlap",
            "description": "A question where 70-90% of content words (nouns, verbs, adjectives) appear in the document, with only minor lexical variations.",
            "probability": 0.167
        },
        {
            "name": "moderate_lexical_overlap",
            "description": "A question where 40-70% of content words appear in the document, with some synonym substitution or rephrasing of key concepts.",
            "probability": 0.167
        },
        {
            "name": "low_lexical_overlap",
            "description": "A question where less than 40% of content words appear in the document, requiring semantic matching rather than lexical matching.",
            "probability": 0.167
        },
        {
            "name": "domain_shift_terminology",
            "description": "A question using terminology from a related but different domain or register (e.g., colloquial vs technical, or terminology from an adjacent field).",
            "probability": 0.166
        },
        {
            "name": "abstraction_level_shift",
            "description": "A question operating at a different abstraction level than the document (e.g., asking about general principles when document discusses specific examples, or vice versa).",
            "probability": 0.166
        }
    ]
}

# ============================================================================
# RQ2: COMPLEMENTARITY CATEGORIZATIONS
# ============================================================================

# Each set uses ONE primary categorization (simplified design)

# Set 1: Question Complexity (coarse)
rq2_set1_complexity = {
    "categorization_name": "question_complexity",
    "categories": [
        {
            "name": "simple",
            "description": "A question that can be answered using information from a single, localized section of the document.",
            "probability": 0.5
        },
        {
            "name": "complex",
            "description": "A question requiring synthesis or connection of information from multiple document sections.",
            "probability": 0.5
        }
    ]
}

# Set 2: Answer Type (coarse)
rq2_set2_answertype = {
    "categorization_name": "question_answertype",
    "categories": [
        {
            "name": "extractive",
            "description": "A question whose answer can be directly extracted from the document without rephrasing.",
            "probability": 0.5
        },
        {
            "name": "abstractive",
            "description": "A question requiring synthesis and generation of new text beyond extraction.",
            "probability": 0.5
        }
    ]
}

# Set 3: Linguistic Variation (coarse)
rq2_set3_vocabulary = {
    "categorization_name": "question_linguisticvariation",
    "categories": [
        {
            "name": "similar_to_document",
            "description": "A question using the same terminology and vocabulary appearing in the document.",
            "probability": 0.5
        },
        {
            "name": "distant_from_document",
            "description": "A question using completely different terminology than the document.",
            "probability": 0.5
        }
    ]
}

# Set 4: Question Phrasing
rq2_set4_phrasing = {
    "categorization_name": "question_phrasing",
    "categories": [
        {
            "name": "concise_and_natural",
            "description": "A question phrased in natural everyday language as a concise direct question of less than 10 words.",
            "probability": 0.25
        },
        {
            "name": "verbose_and_natural",
            "description": "A question phrased in natural everyday language as a relatively long question of more than 9 words.",
            "probability": 0.25
        },
        {
            "name": "short_search_query",
            "description": "A question phrased as a web search query (only keywords, no punctuation or natural structure) of less than 7 words.",
            "probability": 0.25
        },
        {
            "name": "long_search_query",
            "description": "A question phrased as a web search query (only keywords, no punctuation or natural structure) of more than 6 words.",
            "probability": 0.25
        }
    ]
}

# Set 5: User Expertise
# (This is a user categorization, not question categorization)

# ============================================================================
# USER CATEGORIZATIONS (used across all RQs)
# ============================================================================

user_expertise_categorization = {
    "categorization_name": "user_expertise",
    "categories": [
        {
            "name": "expert",
            "description": "A specialized user with deep understanding of technical terminology and domain concepts who expects precise, detailed information.",
            "probability": 0.5
        },
        {
            "name": "novice",
            "description": "A general user with limited domain knowledge who needs explanations using accessible language without specialized jargon.",
            "probability": 0.5
        }
    ]
}

# ============================================================================
# RQ3: INTERACTION ANALYSIS
# ============================================================================

# 2x2 factorial: Linguistic Variation × Question Complexity

rq3_linguisticvariation = {
    "categorization_name": "question_linguisticvariation",
    "categories": [
        {
            "name": "similar_to_document",
            "description": "A question using the same terminology appearing in the document.",
            "probability": 0.5
        },
        {
            "name": "distant_from_document",
            "description": "A question using completely different terminology from the document.",
            "probability": 0.5
        }
    ]
}

rq3_complexity = {
    "categorization_name": "question_complexity",
    "categories": [
        {
            "name": "simple",
            "description": "A question answerable from a single document section.",
            "probability": 0.5
        },
        {
            "name": "complex",
            "description": "A question requiring synthesis across multiple sections.",
            "probability": 0.5
        }
    ]
}