# Taming Hallucinations: A Semantic Matching Evaluation Framework for LLM-Generated Ontologies

Ontology learning using Large Language Models (LLMs) has shown promise, yet remains challenged by hallucinations—spurious or inaccurate concepts and relationships that undermine domain validity. This issue is particularly critical in highly specialized fields such as life sciences, where ontology accuracy directly impacts knowledge representation and decision-making. In this work, we introduce an automated evaluation framework that systematically assesses the quality of LLM-generated ontologies by comparing their concepts and relationship triples against expert-curated domain ontologies. Our approach leverages transformer-based semantic similarity methods to detect inconsistencies, ensuring that generated ontologies align with real-world knowledge. We evaluate our framework using six LLM-generated ontologies, validating them against three reference ontologies with increasing domain specificity. Results demonstrate that our framework significantly enhances ontology reliability, reducing hallucinations while maintaining high semantic alignment with expert knowledge. This work establishes a scalable, automated approach for validating LLM-generated ontologies, paving the way for their broader adoption in complex, knowledge-intensive domains.

This repository provides the code and data for evaluating **LLM-generated ontologies** against **domain reference ontologies**. We aim to identify and “tame” hallucinations—spurious or inaccurate concepts and triples introduced by Large Language Models—through **semantic matching** techniques.

---

## Repository Structure
