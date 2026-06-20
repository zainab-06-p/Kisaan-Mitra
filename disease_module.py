"""
disease_module.py — Crop disease identification, organic treatment, and knowledge base lookup.
"""

import json
import os

from llm import analyze_image_with_vision, call_llm
from prompts import DISEASE_PROMPT_TEMPLATE, SYSTEM_PROMPT_BASE

_KB_PATH = os.path.join(os.path.dirname(__file__), "data", "disease_knowledge.json")


def lookup_disease_info(query: str, image_context: str = "") -> str:
    """
    RAG knowledge retrieval: search the local disease knowledge base for entries
    relevant to the farmer's query and image analysis.

    Uses simple keyword scoring — no embeddings needed at this scale.
    Returns a formatted string of the top matching disease's details,
    or an empty string if nothing relevant is found.
    """
    try:
        with open(_KB_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)
    except Exception as e:
        print(f"[disease_module.py] Could not load knowledge base: {e}")
        return ""

    combined = f"{query} {image_context}".lower()
    scored = []

    for disease in kb.get("diseases", []):
        score = 0

        # Match disease name and aliases
        for kw in disease.get("keywords", []):
            if kw.lower() in combined:
                score += 3

        # Match crop names mentioned in query
        for crop in disease.get("crops", []):
            if crop.lower() in combined:
                score += 2

        # Match symptom words
        for symptom in disease.get("symptoms", []):
            for word in symptom.lower().split():
                if len(word) > 4 and word in combined:
                    score += 1

        if score > 0:
            scored.append((score, disease))

    if not scored:
        return ""

    scored.sort(key=lambda x: -x[0])
    best = scored[0][1]

    # Format a concise but complete knowledge block for the LLM
    remedies_text = ""
    for r in best.get("organic_remedies", []):
        remedies_text += (
            f"  - {r['name']}: {r['preparation']} "
            f"Apply: {r['application']} Frequency: {r['frequency']}\n"
        )

    faqs_text = ""
    for faq in best.get("faqs", []):
        faqs_text += f"  Q: {faq['q']}\n  A: {faq['a']}\n"

    prevention_text = "; ".join(best.get("prevention", []))

    result = (
        f"=== Knowledge Base: {best['name']} ===\n"
        f"Cause: {best.get('cause', '')}\n"
        f"Conditions that worsen it: {best.get('favourable_conditions', '')}\n"
        f"Key symptoms: {'; '.join(best.get('symptoms', [])[:4])}\n"
        f"Organic remedies:\n{remedies_text}"
        f"Prevention: {prevention_text}\n"
        f"Common farmer questions answered:\n{faqs_text}"
        f"When to visit KVK: {best.get('when_to_consult_kvk', '')}"
    )
    print(f"[disease_module.py] KB match: {best['name']} (score={scored[0][0]})")
    return result


def handle_disease_query(text_query: str, image_path: str = None) -> str:
    """
    Handle a crop disease query from the farmer.

    Steps:
    1. If an image is provided, run it through the vision model to extract symptoms.
    2. Combine the image analysis with the farmer's text description.
    3. Fill the DISEASE_PROMPT_TEMPLATE and call the LLM for organic remedies.
    4. Return the LLM response.

    Args:
        text_query:  The farmer's spoken or typed description of the problem.
        image_path:  Optional path to an uploaded crop photo.

    Returns:
        A string containing the disease identification and organic treatment advice.
    """
    image_analysis = ""
    if image_path:
        print(f"[disease_module.py] Analyzing image: {image_path}")
        image_analysis = analyze_image_with_vision(image_path, description=text_query)
        if image_analysis:
            print(f"[disease_module.py] Vision analysis: {image_analysis[:200]}...")

    # Build the image analysis section for the prompt
    if image_analysis:
        image_analysis_section = (
            f"\nImage analysis from photo provided by farmer:\n{image_analysis}"
        )
        combined_description = f"{text_query}\n\n[Photo analysis: {image_analysis}]"
    else:
        image_analysis_section = ""
        combined_description = text_query

    filled_prompt = DISEASE_PROMPT_TEMPLATE.format(
        disease_description=combined_description,
        image_analysis_section=image_analysis_section,
    )

    response = call_llm(SYSTEM_PROMPT_BASE, filled_prompt)
    return response
