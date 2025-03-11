

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import json

# Load CSV file
csv_file = "/content/drive/MyDrive/AquaDiva/ENVO.csv"  # Change this to your actual file path
df = pd.read_csv(csv_file)

# Extract relevant columns
ontology_dict = {}
missing_class_label = []
missing_both = []

for index, row in df.iterrows():
    concept = str(row["classLabel"]).strip() if pd.notna(row["classLabel"]) else None
    definition = str(row["classComment"]).strip() if pd.notna(row["classComment"]) else ""

    if concept:  # If classLabel exists, add to dictionary
        ontology_dict[concept] = definition
    else:
        if pd.isna(row["classComment"]):
            missing_both.append(index)  # Row where both are missing
        else:
            missing_class_label.append(index)  # Row where only classLabel is missing

# Save to JSON
output_json = "/content/drive/MyDrive/AquaDiva/ENVO_concepts.json"
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(ontology_dict, f, indent=4, ensure_ascii=False)

# Print missing rows
print(f"✅ Ontology JSON saved as {output_json}")
print(f"⚠️ Rows with missing 'classLabel': {missing_class_label}")
print(f"⚠️ Rows with both 'classLabel' and 'classComment' missing: {missing_both}")

!pip install rapidfuzz

import json
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz

# Load Pretrained Sentence Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

import json
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA

# Load Pretrained Sentence Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

### 1. Load Ontologies ###
def load_ontology(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load LLM-generated ontology
llm_ontology = load_ontology("/content/Experiment6CarbonNitrogen.json")

# Load two domain ontologies
domain_ontology_1 = load_ontology("/content/drive/MyDrive/AquaDiva/ENVO_concepts.json")  # ENVO
domain_ontology_2 = load_ontology("/content/drive/MyDrive/AquaDiva/INBIO_concepts.json")  # INBIO

# Combine both ontologies into one dictionary
combined_domain_ontology = {**domain_ontology_1, **domain_ontology_2}  # Merges both ontologies

### 2. Embed Concepts + Definitions ###
def embed_text(text):
    """Generate an embedding for a given text."""
    return embedding_model.encode(text, convert_to_tensor=True)

# Create embeddings for LLM-generated concepts
llm_embeddings = {concept: embed_text(concept + " - " + definition) for concept, definition in llm_ontology.items()}

# Create embeddings for domain ontologies separately
domain_embeddings_1 = {concept: embed_text(concept + " - " + definition) for concept, definition in domain_ontology_1.items()}
domain_embeddings_2 = {concept: embed_text(concept + " - " + definition) for concept, definition in domain_ontology_2.items()}

### 3. Compare Embeddings & Filter Concepts ###
accepted_matches = []
hallucinated_concepts = []
similarity_threshold = 0.55  # Adjust if needed
matched_in_envo = 0
matched_in_inbio = 0

for llm_concept, llm_emb in llm_embeddings.items():
    best_match = None
    best_score = 0
    best_ontology = None

    # Check in Ontology 1 (ENVO)
    for domain_concept, domain_emb in domain_embeddings_1.items():
        score = util.cos_sim(llm_emb, domain_emb).item()
        if score > best_score:
            best_score = score
            best_match = domain_concept
            best_ontology = "ENVO (Ontology 1)"

    # Check in Ontology 2 (INBIO)
    for domain_concept, domain_emb in domain_embeddings_2.items():
        score = util.cos_sim(llm_emb, domain_emb).item()
        if score > best_score:
            best_score = score
            best_match = domain_concept
            best_ontology = "INBIO (Ontology 2)"

    if best_score >= similarity_threshold:
        accepted_matches.append((llm_concept, best_match, best_ontology, best_score))
        if best_ontology == "ENVO (Ontology 1)":
            matched_in_envo += 1
        elif best_ontology == "INBIO (Ontology 2)":
            matched_in_inbio += 1
        print(f"✅ Keeping '{llm_concept}' (Matched with '{best_match}' in {best_ontology}, Similarity: {best_score:.2f})")
    else:
        hallucinated_concepts.append(llm_concept)
        print(f"❌ Removing '{llm_concept}' (No strong match found, Best Similarity: {best_score:.2f})")

### 4. Print Statistics ###
total_llm_concepts = len(llm_ontology)
matched_concepts = len(accepted_matches)
hallucinated_concepts_count = len(hallucinated_concepts)

percentage_matched = (matched_concepts / total_llm_concepts) * 100
percentage_hallucinated = (hallucinated_concepts_count / total_llm_concepts) * 100

print("\n📊 **Matching Statistics**:")
print(f"✅ Matched Concepts: {matched_concepts} / {total_llm_concepts} ({percentage_matched:.2f}%)")
print(f"❌ Hallucinated Concepts: {hallucinated_concepts_count} / {total_llm_concepts} ({percentage_hallucinated:.2f}%)")
print(f"🌿 Matched in ENVO (Ontology 1): {matched_in_envo}")
print(f"🔬 Matched in INBIO (Ontology 2): {matched_in_inbio}")

### 5. Prepare Data for Plotting ###
llm_emb_list = [llm_embeddings[c1].cpu().numpy() for c1, _, _, _ in accepted_matches]
domain_emb_list = [
    (domain_embeddings_1[c2].cpu().numpy() if best_ontology == "ENVO (Ontology 1)" else domain_embeddings_2[c2].cpu().numpy())
    for _, c2, best_ontology, _ in accepted_matches
]

# Reduce to 2D using PCA
all_embeddings = np.vstack(llm_emb_list + domain_emb_list)
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(all_embeddings)

# Split reduced embeddings into LLM and domain ontology points
llm_points = reduced_embeddings[:len(accepted_matches)]
domain_points = reduced_embeddings[len(accepted_matches):]

plt.figure(figsize=(10, 6))

# Scatter plots for the legend (only plotted once to avoid duplication)
plt.scatter([], [], color='red', label="LLM Concept")
plt.scatter([], [], color='green', label="ENVO")
plt.scatter([], [], color='blue', label="INBIO")

# Plot the actual data
for i, (c1, c2, best_ontology, score) in enumerate(accepted_matches):
    color = 'green' if best_ontology == "ENVO (Ontology 1)" else 'blue'  # Green for ENVO, Blue for INBIO

    plt.scatter(llm_points[i, 0], llm_points[i, 1], color='red')  # LLM Concept
    plt.scatter(domain_points[i, 0], domain_points[i, 1], color=color)  # Matched ontology concept
    plt.plot([llm_points[i, 0], domain_points[i, 0]], [llm_points[i, 1], domain_points[i, 1]], 'k--', alpha=0.5)
    plt.text(llm_points[i, 0], llm_points[i, 1], c1, fontsize=10, color='red')
    plt.text(domain_points[i, 0], domain_points[i, 1], c2, fontsize=10, color=color)

# Fix title and labels
plt.legend()
plt.title(f"Concept Matching: LLM-Carbon vs INBIO & ENVO | {percentage_matched:.2f}% Matched")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)

# Save figure
plt.savefig("/content/Carbonconcept_matching_plot.png", dpi=300, bbox_inches='tight')

plt.show()

hallucinated_concepts

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# Load pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to load JSON file
def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Function to generate embeddings for concepts and definitions
def generate_embeddings(concepts_dict):
    embeddings = {}
    for concept, definition in concepts_dict.items():
        text = f"{concept}: {definition}".strip()
        embeddings[concept] = model.encode(text)
    return embeddings

# Function to compute cosine similarity between concept embeddings
def compute_similarity(embeddings1, embeddings2):
    concepts1 = list(embeddings1.keys())
    concepts2 = list(embeddings2.keys())
    matrix1 = np.array(list(embeddings1.values()))
    matrix2 = np.array(list(embeddings2.values()))
    similarity_matrix = cosine_similarity(matrix1, matrix2)
    return similarity_matrix, concepts1, concepts2

# Function to summarize matching results
def summarize_matches(similarity_matrix, concepts1, concepts2, threshold=0.55):
    matched = 0
    unmatched = []
    matches_per_ontology = {}
    for i, concept in enumerate(concepts1):
        max_sim = max(similarity_matrix[i])
        if max_sim >= threshold:
            matched += 1
            match_index = np.argmax(similarity_matrix[i])
            matched_concept = concepts2[match_index]
            matches_per_ontology[concept] = matched_concept
        else:
            unmatched.append(concept)
    return matched, unmatched, matches_per_ontology

# Function to plot PCA visualization of concept similarity
def plot_pca(embeddings_dict, matches_per_ontology):
    all_concepts = list(embeddings_dict.keys())
    all_embeddings = np.array(list(embeddings_dict.values()))

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(all_embeddings)

    plt.figure(figsize=(12, 8))

    colors = {"JSON": "red", "ENVO": "green", "OBOE_SBC": "blue"}
    markers = {"JSON": "o", "ENVO": "s", "OBOE_SBC": "^"}

    for i, concept in enumerate(all_concepts):
        category = "JSON" if concept in matches_per_ontology else "ENVO"
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], color=colors[category], marker=markers[category], label=concept if category == "JSON" else "")
        plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], concept, fontsize=8, color=colors[category])

    for json_concept, matched_concept in matches_per_ontology.items():
        json_idx = all_concepts.index(json_concept)
        matched_idx = all_concepts.index(matched_concept)
        plt.plot([reduced_embeddings[json_idx, 0], reduced_embeddings[matched_idx, 0]],
                 [reduced_embeddings[json_idx, 1], reduced_embeddings[matched_idx, 1]], "k--", alpha=0.5)

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Concept Matching: AquaDivaVersion3 vs ENVO & OBOE_SBC")
    plt.legend(["LLM Concept", "ENVO", "OBOE_SBC"], loc="upper left")
    plt.show()

# Main execution
if __name__ == "__main__":
    json_path = "/content/carbon.json"  # Replace with actual JSON file
    json_concepts = load_json(json_path)

    # Generate embeddings for JSON concepts
    json_embeddings = generate_embeddings(json_concepts)

    # Load ontology concepts (replace with actual ontology data sources)
    ontology_paths = ["/content/envo_classes.json", "/content/oboe_sbc_classes.json"]
    ontology_embeddings = {}
    for path in ontology_paths:
        ontology_data = load_json(path)
        ontology_embeddings[path] = generate_embeddings(ontology_data)

    # Compute similarity and summarize results
    results = []
    all_embeddings_dict = json_embeddings.copy()

    for ontology_name, embeddings in ontology_embeddings.items():
        similarity_matrix, concepts1, concepts2 = compute_similarity(json_embeddings, embeddings)
        matched, unmatched, matches_per_ontology = summarize_matches(similarity_matrix, concepts1, concepts2)
        results.append([ontology_name, matched, len(unmatched)])
        all_embeddings_dict.update(embeddings)

        # Print unmatched concepts
        print(f"\nUnmatched concepts in {ontology_name}:")
        for concept in unmatched:
            print(concept)

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results, columns=["Ontology", "Matched", "Unmatched"])
    print(df_results)

# Generate PCA plot
plot_pca(all_embeddings_dict, matches_per_ontology)

from google.colab import drive
drive.mount('/content/drive')

import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_embeddings(concepts, model, tokenizer):
    texts = [f"{concept} {tokenizer.sep_token} {definition}" for concept, definition in concepts.items()]
    return model.encode(texts, convert_to_numpy=True)

def match_concepts(llm_concepts, domain_ontologies, model, tokenizer, threshold=0.55):
    total_llm_concepts = len(llm_concepts)
    unmatched_concepts = llm_concepts.copy()
    print(f"Total LLM Concepts: {len(unmatched_concepts)}")
    matched_results = []
    summary_results = []

    for i, (ontology_name, domain_concepts) in enumerate(domain_ontologies.items()):
        print(f"Processing {ontology_name}...")

        llm_embeddings = compute_embeddings(unmatched_concepts, model, tokenizer)
        domain_embeddings = compute_embeddings(domain_concepts, model, tokenizer)

        similarities = cosine_similarity(llm_embeddings, domain_embeddings)
        matched_indices = np.where(similarities >= threshold)

        matched_concepts = {}
        for llm_idx, domain_idx in zip(*matched_indices):
            llm_concept = list(unmatched_concepts.keys())[llm_idx]
            domain_concept = list(domain_concepts.keys())[domain_idx]
            matched_concepts[llm_concept] = domain_concept
            matched_results.append(f"{llm_concept} -> {domain_concept} ({ontology_name})")

        # Remove matched concepts from unmatched list
        unmatched_concepts = {k: v for k, v in unmatched_concepts.items() if k not in matched_concepts}
        matched_percentage = ((total_llm_concepts- len(unmatched_concepts)) * 100 ) / total_llm_concepts

        unmatched_list = list(unmatched_concepts.keys())
        summary_results.append((ontology_name, matched_percentage, unmatched_list))

        # Save unmatched concepts after each ontology match
        unmatched_df = pd.DataFrame({"Unmatched Concepts": unmatched_list})
        unmatched_csv = f"unmatched_concepts_after_{ontology_name}.csv"
        unmatched_df.to_csv(unmatched_csv, index=False)
        print(f"Unmatched concepts after {ontology_name} saved to {unmatched_csv}")


    return summary_results, unmatched_concepts, matched_results

def save_matched_results(matched_results, file_path='matched_concepts.txt'):
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in matched_results:
            f.write(line + "\n")
    print(f"Matched concepts saved to {file_path}")

def main(llm_file, ontology_files):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    tokenizer = model.tokenizer
    llm_concepts = load_json(llm_file)
    domain_ontologies = {f'Ontology {i+1}': load_json(file) for i, file in enumerate(ontology_files)}

    summary_results, unmatched_concepts, matched_results = match_concepts(llm_concepts, domain_ontologies, model, tokenizer)
    save_matched_results(matched_results)

    # Display results in a dataframe
    df = pd.DataFrame(summary_results, columns=["Ontology", "% of Matched Concepts", "Unmatched Concepts"])
    print(df)  # Display in console
    df.to_csv("semantic_matching_results.csv", index=False)  # Save results to a CSV file
    df.to_csv("/content/drive/MyDrive/AquaDiva/LLMConcepts/AquaDivaOntologyVersion1_semantic_matching_results.csv", index=False)  # Save results to a CSV file
    print("Results saved to semantic_matching_results.csv")
    print("Processing complete!")


    llm_file = '/content/drive/MyDrive/AquaDiva/LLMConcepts/AquaDivaOntologyVersion1.json'
#ontology_files = ['/content/drive/MyDrive/AquaDiva/BioPortalConcepts/oboe_sbc_classes.json', '/content/drive/MyDrive/AquaDiva/BioPortalConcepts/envo_classes.json', '/content/drive/MyDrive/AquaDiva/BioPortalConcepts/chebi_classes.json']
    ontology_files = ['/content/drive/MyDrive/AquaDiva/BioPortalConcepts/oboe_sbc_classes.json', '/content/drive/MyDrive/AquaDiva/BioPortalConcepts/envo_classes.json', '/content/drive/MyDrive/AquaDiva/BioPortalConcepts/chebi_classes.json']

    main(llm_file, ontology_files)

llm_file = '/content/drive/MyDrive/AquaDiva/LLMConcepts/AquaDivaOntologyVersion1.json'
#ontology_files = ['/content/drive/MyDrive/AquaDiva/BioPortalConcepts/oboe_sbc_classes.json', '/content/drive/MyDrive/AquaDiva/BioPortalConcepts/envo_classes.json', '/content/drive/MyDrive/AquaDiva/BioPortalConcepts/chebi_classes.json']
ontology_files = ['/content/drive/MyDrive/AquaDiva/BioPortalConcepts/oboe_sbc_classes.json', '/content/drive/MyDrive/AquaDiva/BioPortalConcepts/envo_classes.json', '/content/drive/MyDrive/AquaDiva/BioPortalConcepts/chebi_classes.json']

main(llm_file, ontology_files)

print("hh")

from sentence_transformers import SentenceTransformer, util

# Load the pre-trained model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Define two ontology triples
#triple1 = "KarstGroundwater is a Water"
#triple2 = "freshwater subclass of water"
triple1 = "TraceGas is_consumed_by MicrobialCommunity"
triple2 = "methane has role bacterial metabolite"
# Convert triples to embeddings
embedding1 = model.encode(triple1, convert_to_tensor=True)
embedding2 = model.encode(triple2, convert_to_tensor=True)

# Compute cosine similarity
similarity_score = util.cos_sim(embedding1, embedding2).item()

# Print similarity score
print(f"Similarity between triples: {similarity_score:.4f}")

triple1 = "{TraceGas} {is_consumed_by} {MicrobialCommunity}"
triple2 = "{methane} {has role} {bacterial metabolite}"
# Convert triples to embeddings
embedding1 = model.encode(triple1, convert_to_tensor=True)
embedding2 = model.encode(triple2, convert_to_tensor=True)

# Compute cosine similarity
similarity_score = util.cos_sim(embedding1, embedding2).item()

# Print similarity score
print(f"Similarity between triples: {similarity_score:.4f}")

triple1 = "{KarstGroundwater} {is a} {Water}"
triple2 = "{freshwater} {subclass of} {water}"
# Convert triples to embeddings
embedding1 = model.encode(triple1, convert_to_tensor=True)
embedding2 = model.encode(triple2, convert_to_tensor=True)

# Compute cosine similarity
similarity_score = util.cos_sim(embedding1, embedding2).item()

# Print similarity score
print(f"Similarity between triples: {similarity_score:.4f}")

