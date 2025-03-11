# Taming Hallucinations: A Semantic Matching Evaluation Framework for LLM-Generated Ontologies

Ontology learning using Large Language Models (LLMs) has shown promise, yet remains challenged by hallucinations—spurious or inaccurate concepts and relationships that undermine domain validity. This issue is particularly critical in highly specialized fields such as life sciences, where ontology accuracy directly impacts knowledge representation and decision-making. In this work, we introduce an automated evaluation framework that systematically assesses the quality of LLM-generated ontologies by comparing their concepts and relationship triples against expert-curated domain ontologies. Our approach leverages transformer-based semantic similarity methods to detect inconsistencies, ensuring that generated ontologies align with real-world knowledge. We evaluate our framework using six LLM-generated ontologies, validating them against three reference ontologies with increasing domain specificity. Results demonstrate that our framework significantly enhances ontology reliability, reducing hallucinations while maintaining high semantic alignment with expert knowledge. This work establishes a scalable, automated approach for validating LLM-generated ontologies, paving the way for their broader adoption in complex, knowledge-intensive domains.

This repository provides the code and data for evaluating **LLM-generated ontologies** against **domain reference ontologies**. We aim to identify and “tame” hallucinations—spurious or inaccurate concepts and triples introduced by Large Language Models—through **semantic matching** techniques.

---

## Repository Structure

### `data/` Directory
- **`bioportal_concepts/` & `bioportal_triples/`**  
  Reference ontology data (e.g., exported from BioPortal).
- **`llm_concepts/` & `llm_triples/`**  
  Concepts and triples extracted from LLM-generated ontologies.

### `results/` Directory
- **`concept_matching_summary/`**  
  CSV summaries of **concept-level** semantic matching results (e.g., `<Ontology>_semantic_matching_results.csv`).
- **`matched_triples/`**  
  Text files listing the **matched** Subject–Predicate–Object (SPO) triples per experiment.
- **`triple_matching_summary/`**  
  CSV summaries of **triple-level** semantic matching results.

---

## Python Scripts

1. **`extract_ontology_concepts.py`**  
   Parses ontology files (LLM or reference) to extract concept labels and definitions, then saves them in a structured format.

2. **`extract_ontology_triples.py`**  
   Extracts SPO relationships (e.g., `subClassOf`, `rdf:type`) from ontology files and stores them as CSV or JSON.

3. **`ontology_concept_matching.py`**  
   Loads extracted concepts (LLM vs. reference), uses a sentence-transformer to compute similarity scores, and outputs matched vs. unmatched concepts.

4. **`ontology_triple_matching.py`**  
   Converts SPO triples into sentence-like form, embeds them, and compares to reference triples to identify matches or hallucinations.

---

## Basic Usage

```bash
python ontology_concept_matching.py --llm_concepts results/llm_concepts.json \
                                    --ref_concepts results/bioportal_concepts.json \
                                    --output results/concept_matching_summary/...

python ontology_triple_matching.py --llm_triples results/llm_triples.csv \
                                   --ref_triples results/bioportal_triples.csv \
                                   --output results/triple_matching_summary/...
