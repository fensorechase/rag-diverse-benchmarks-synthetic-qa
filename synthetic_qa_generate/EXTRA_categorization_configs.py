# EXTRA_categorization_configs.py
# Generate ONLY the new categories to complete the hierarchy

# ============================================================================
# Question Complexity: 2 NEW categories only
# ============================================================================
question_complexity_NEW_ONLY = {
    "categorization_name": "question_complexity",
    "categories": [
        {
            "name": "entity_extraction",
            "description": "A question asking for a specific named entity (person, place, organization, date, number) that appears as a single, easily identifiable term in the document.",
            "probability": 0.5
        },
        {
            "name": "comparative_synthesis_concepts",
            "description": "A question requiring comparison of abstract concepts, methodologies, approaches, or theoretical frameworks discussed in the document (e.g., comparing philosophies, strategies, or perspectives).",
            "probability": 0.5
        }
    ]
}

# ============================================================================
# Answer Type: 1 NEW category only
# ============================================================================
question_answertype_NEW_ONLY = {
    "categorization_name": "question_answertype",
    "categories": [
        {
            "name": "explanatory_synthesis",
            "description": "A question requiring explanation of how or why something works, happens, or is true, based on synthesizing multiple pieces of information from the document into a coherent explanation.",
            "probability": 1.0
        }
    ]
}

# ============================================================================
# Linguistic Variation: 2 NEW categories only
# ============================================================================
question_linguisticvariation_NEW_ONLY = {
    "categorization_name": "question_linguisticvariation",
    "categories": [
        {
            "name": "moderate_low_lexical_overlap",
            "description": "A question where 30-50% of content words appear in the document, requiring partial semantic matching as lexical matching becomes insufficient.",
            "probability": 0.5
        },
        {
            "name": "synonym_based_rephrase",
            "description": "A question where most key terms are systematically replaced with direct synonyms or closely related terms, maintaining similar semantic level but different vocabulary.",
            "probability": 0.5
        }
    ]
}

# ============================================================================
# User categorization (same as before)
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


# TOTAL: 112 + 48 + 112 = 272 new questions
