import json
import re
import spacy
import pandas as pd

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load domain knowledge from JSON file
with open("domain_knowledge.json") as f:
    knowledge = json.load(f)

# Function to extract known entities using dictionary lookup
def extract_entities(text, knowledge):
    entities = {
        "competitors": [],
        "features": [],
        "pricing_keywords": []
    }

    for category, keywords in knowledge.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                entities[category].append(keyword)

    return entities

# Function to extract named entities using spaCy
def extract_ner_entities(text):
    doc = nlp(text)
    ner_entities = {"orgs": [], "persons": [], "locations": []}

    for ent in doc.ents:
        if ent.label_ == "ORG":
            ner_entities["orgs"].append(ent.text)
        elif ent.label_ == "PERSON":
            ner_entities["persons"].append(ent.text)
        elif ent.label_ in ["GPE", "LOC"]:
            ner_entities["locations"].append(ent.text)

    return ner_entities

# Combined function for entity extraction
def extract_combined_entities(text):
    dict_entities = extract_entities(text, knowledge)
    ner_entities = extract_ner_entities(text)

    combined_entities = {
        "competitors": dict_entities["competitors"] + ner_entities["orgs"],
        "features": dict_entities["features"],
        "pricing_keywords": dict_entities["pricing_keywords"],
        "persons": ner_entities["persons"],
        "locations": ner_entities["locations"]
    }

    return combined_entities

if __name__ == "__main__":
    # Test Example
    sample_text = "We offer an AI engine with a flexible pricing model and scalable customer support to Tesla Motors founded by Elon Musk."
    extracted_entities = extract_combined_entities(sample_text)
    print(json.dumps(extracted_entities, indent=4))
