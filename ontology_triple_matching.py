

from google.colab import drive
drive.mount('/content/drive')

import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

# Load triples from CSV file
def load_triples(csv_file):
    df = pd.read_csv(csv_file)
    triples = df[["Subject", "Predicate", "Object"]].values.tolist()
    return triples

# Convert triples into sentences
def convert_triples_to_sentences(triples):
    return [f"{s} {p} {o}" for s, p, o in triples]

# Compute embeddings for sentences
def compute_embeddings(sentences, model):
    return model.encode(sentences, convert_to_numpy=True)

# Match triples across multiple ontologies
def match_triples(main_triples, domain_ontologies, model, threshold=0.5):
    total_main_triples = len(main_triples)
    unmatched_triples = main_triples.copy()
    print(f"Total Main Ontology Triples: {total_main_triples}")

    matched_results = []
    summary_results = []

    # Convert main triples to sentences & compute embeddings
    main_sentences = convert_triples_to_sentences(main_triples)
    main_embeddings = compute_embeddings(main_sentences, model)

    for ontology_name, domain_triples in domain_ontologies.items():
        print(f"Processing {ontology_name}...")

        domain_sentences = convert_triples_to_sentences(domain_triples)
        domain_embeddings = compute_embeddings(domain_sentences, model)

        # Compute similarity between main triples and domain ontology triples
        similarities = util.pytorch_cos_sim(main_embeddings, domain_embeddings)
        matched_indices = np.where(similarities >= threshold)

        matched_triples = {}
        for main_idx, domain_idx in zip(*matched_indices):
            main_triple = main_sentences[main_idx]
            domain_triple = domain_sentences[domain_idx]
            matched_triples[main_triple] = domain_triple
            matched_results.append(f"{main_triple} -> {domain_triple} ({ontology_name})")

        # Remove matched triples from unmatched list
        unmatched_triples = [t for t in unmatched_triples if convert_triples_to_sentences([t])[0] not in matched_triples]
        matched_percentage = ((total_main_triples - len(unmatched_triples)) * 100) / total_main_triples

        unmatched_list = convert_triples_to_sentences(unmatched_triples)
        summary_results.append((ontology_name, matched_percentage, unmatched_list))

        # Save unmatched triples after each ontology match
        unmatched_df = pd.DataFrame({"Unmatched Triples": unmatched_list})
        unmatched_csv = f"/content/drive/MyDrive/AquaDiva/TripleMatching/Experiment6CarbonNitrogen_unmatched_triples_after_{ontology_name}.csv"
        unmatched_df.to_csv(unmatched_csv, index=False)
        print(f"Unmatched triples after {ontology_name} saved to {unmatched_csv}")

    return summary_results, unmatched_triples, matched_results

# Save matched triples
def save_matched_results(matched_results, file_path="/content/drive/MyDrive/AquaDiva/TripleMatching/Experiment6CarbonNitrogen_matched_triples.txt"):
    with open(file_path, "w", encoding="utf-8") as f:
        for line in matched_results:
            f.write(line + "\n")
    print(f"✅ Matched triples saved to {file_path}")

# Main function
def main(main_csv, ontology_csv_files):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    main_triples = load_triples(main_csv)
    domain_ontologies = {f"Ontology {i+1}": load_triples(file) for i, file in enumerate(ontology_csv_files)}

    summary_results, unmatched_triples, matched_results = match_triples(main_triples, domain_ontologies, model)

    save_matched_results(matched_results)

    # Save summary as CSV
    summary_df = pd.DataFrame(summary_results, columns=["Ontology", "% Matched", "Unmatched Triples"])
    summary_csv = "/content/drive/MyDrive/AquaDiva/TripleMatching/Experiment6CarbonNitrogen_semantic_matching_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n✅ Summary saved to:", summary_csv)
    print("\n✅ Processing complete!")

# Example Usage
main_csv = "/content/drive/MyDrive/AquaDiva/LLMTriples/Experiment6CarbonNitrogen_cleaned.csv"

  # Main ontology triples
ontology_csv_files = [
    "/content/drive/MyDrive/AquaDiva/BioPortalTriples/Triples/OBOE-SBC_cleaned.csv",
    "/content/drive/MyDrive/AquaDiva/BioPortalTriples/Triples/ENVO_cleaned_FinalFinal.csv",
    "/content/drive/MyDrive/AquaDiva/BioPortalTriples/Triples/chebi_triples_cleaned.csv"
]

main(main_csv, ontology_csv_files)

