

!git clone https://github.com/NadeenAhmad/NeOn-GPTAquaDivaOntology

!pip install rdflib

"""## Extract LLM-ontology Triples"""

from rdflib import Graph, RDF, RDFS, OWL, BNode
import os
import csv

def format_node(node):
    if isinstance(node, BNode):  # Skip blank nodes
        return None
    return node.split("#")[-1] if "#" in node else node.split("/")[-1]

def extract_spo_triples(ontology_url, format=None):
    g = Graph()

    # Determine format if not explicitly provided
    if format is None:
        if ontology_url.endswith(".ttl"):
            format = "turtle"
        elif ontology_url.endswith(".rdf") or ontology_url.endswith(".owl"):
            format = "application/rdf+xml"
        else:
            raise ValueError("Unsupported ontology format. Please specify 'format' explicitly.")

    g.parse(ontology_url, format=format)  # Load ontology

    spo_triples = []

    # Dictionary to store multiple domains and ranges for each predicate
    predicate_info = {}

    # Step 1: Extract domain and range information
    for s, p, o in g:
        if isinstance(s, BNode) or isinstance(o, BNode):
            continue  # Skip blank nodes
        if p == RDFS.domain:
            predicate_info.setdefault(s, {"domain": set(), "range": set()})["domain"].add(o)
        elif p == RDFS.range:
            predicate_info.setdefault(s, {"domain": set(), "range": set()})["range"].add(o)

    # Step 2: Construct SPO triples for properties with domain and range
    for predicate, info in predicate_info.items():
        for domain in info["domain"]:
            for range_ in info["range"]:
                subject = format_node(domain)
                pred = format_node(predicate)
                obj = format_node(range_)
                if subject and obj:  # Ensure blank nodes are skipped
                    spo_triples.append((subject, pred, obj))

    # Step 3: Extract subclass relationships
    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
        subject = format_node(s)
        predicate = "subClassOf"
        obj = format_node(o)
        if subject and obj:  # Ensure blank nodes are skipped
            spo_triples.append((subject, predicate, obj))

    # Step 4: Extract "is a" (rdf:type) relationships
    for s, p, o in g.triples((None, RDF.type, None)):
        subject = format_node(s)
        predicate = "is a"
        obj = format_node(o)
        if subject and obj:  # Ensure blank nodes are skipped
            spo_triples.append((subject, predicate, obj))

    return spo_triples

def save_triples_to_csv(file, triples, filename="triples.csv"):
    filename2 = "/content/drive/MyDrive/AquaDiva/LLMTriples/"+file+"_"+filename
    with open(filename2, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Subject", "Predicate", "Object"])
        writer.writerows(triples)

if __name__ == "__main__":
    url = "Experiment3AquaDiva"
    ontology_url = "/content/NeOn-GPTAquaDivaOntology/Results/"+url+".ttl"  # Replace with the actual URL or file path
    triples = extract_spo_triples(ontology_url)
    save_triples_to_csv(url, triples)

    print("\nExtracted Subject-Predicate-Object Triples:")
    for subject, predicate, obj in triples:
        print(f"({subject}) -[{predicate}]-> ({obj})")
    print("\nExtracted Subject-Predicate-Object Triples saved to triples.csv")

from google.colab import drive
drive.mount('/content/drive')

"""## Extract BioPortal Triples"""

import rdflib
import csv

# Load the ontology from BioPortal
ontology_name = "OBOE-SBC"  # Extracted ontology name
csv_filename = f"{ontology_name}.csv"  # Filename based on ontology

ontology_url = "https://data.bioontology.org/ontologies/OBOE-SBC/submissions/1/download?apikey=87918894-263d-4f71-9b22-bfc345c1800d"

# Initialize RDF Graph
g = rdflib.Graph()
g.parse(ontology_url, format="application/rdf+xml")

# Extract all properties dynamically by finding all predicates in triples
all_properties = set(g.predicates())

# Extract all triples where these properties are used
triples = []
for prop in all_properties:
    for s, _, o in g.triples((None, prop, None)):
        triples.append((s, prop, o))

# Function to extract local names from URIs
def get_local_name(uri):
    if isinstance(uri, rdflib.term.URIRef):
        return uri.split("#")[-1] if "#" in uri else uri.split("/")[-1]
    elif isinstance(uri, rdflib.term.Literal):
        return str(uri)
    return uri  # Return as-is if not a URI or Literal

# Process triples into a structured format
formatted_triples = []
for s, p, o in triples:
    formatted_triples.append([get_local_name(s), get_local_name(p), get_local_name(o)])

# Save formatted triples to a CSV file
csv_filepath = f"/content/drive/MyDrive/AquaDiva/BioPortalTriples/{csv_filename}"  # Save to a persistent path
with open(csv_filepath, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Subject", "Predicate", "Object"])  # CSV header
    writer.writerows(formatted_triples)

print(f"\n✅ Data saved to {csv_filename} successfully!")

import rdflib
import csv

# Define ontology details
ontology_name = "ENVO"  # Ontology name
csv_filename = f"{ontology_name}.csv"  # Filename based on ontology

# Path to the downloaded OWL file (update this if needed)
local_owl_file = "/content/envo2.owl"  # Change this to your actual file path

# Initialize RDF Graph
g = rdflib.Graph()

# Load the ontology from the local OWL file
g.parse(local_owl_file, format="xml")  # OWL files are typically XML-based

# Extract all properties dynamically by finding all predicates in triples
all_properties = set(g.predicates())

# Extract all triples where these properties are used
triples = []
for prop in all_properties:
    for s, _, o in g.triples((None, prop, None)):
        triples.append((s, prop, o))

# Function to extract local names from URIs
def get_local_name(uri):
    if isinstance(uri, rdflib.term.URIRef):
        return uri.split("#")[-1] if "#" in uri else uri.split("/")[-1]
    elif isinstance(uri, rdflib.term.Literal):
        return str(uri)
    return uri  # Return as-is if not a URI or Literal

# Process triples into a structured format
formatted_triples = []
for s, p, o in triples:
    formatted_triples.append([get_local_name(s), get_local_name(p), get_local_name(o)])

# Save formatted triples to a CSV file
csv_filepath = f"/content/drive/MyDrive/AquaDiva/BioPortalTriples/{csv_filename}"  # Save to a persistent path
with open(csv_filepath, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Subject", "Predicate", "Object"])  # CSV header
    writer.writerows(formatted_triples)

print(f"\n✅ Data saved to {csv_filename} successfully!")

"""## Labeled Ontology"""

import rdflib
import csv

# Define ontology details
ontology_name = "ENVO"  # Update as needed
csv_filename = f"{ontology_name}_labeled.csv"

# Path to the OWL or OBO file (update this!)
ontology_file = "/content/envo2.owl"  # Change for your file

# Initialize RDF Graph
g = rdflib.Graph()

# Load ontology from OWL file (for OBO, use 'pronto' instead)
g.parse(ontology_file, format="xml")

# Function to get labels from URIs
def get_label(uri):
    label = None
    if isinstance(uri, rdflib.term.URIRef):
        # Try fetching the rdfs:label for this entity
        for _, _, label_value in g.triples((uri, rdflib.RDFS.label, None)):
            label = str(label_value)
            break
        return label if label else uri.split("#")[-1] if "#" in uri else uri.split("/")[-1]
    elif isinstance(uri, rdflib.term.Literal):
        return str(uri)
    return uri  # Return as-is if not URI or Literal

# Extract all predicates and their triples
triples = []
for s, p, o in g:
    subject_label = get_label(s)
    predicate_label = get_label(p)
    object_label = get_label(o)

    # Append only if labels exist
    if subject_label and predicate_label and object_label:
        triples.append([subject_label, predicate_label, object_label])

# Save the improved triples to a CSV file
csv_filepath = f"/content/drive/MyDrive/AquaDiva/BioPortalTriples/{csv_filename}"
with open(csv_filepath, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Subject", "Predicate", "Object"])  # CSV header
    writer.writerows(triples)

print(f"\n✅ Data saved to {csv_filename} successfully!")

"""## Cleaned Ontology"""

import pandas as pd
import rdflib
import requests
import re

# Load labeled triples CSV
csv_filepath = "/content/drive/MyDrive/AquaDiva/BioPortalTriples/ENVO_cleaned_Final.csv"  # Input file
output_csv_filepath = "/content/drive/MyDrive/AquaDiva/BioPortalTriples/ENVO_cleaned_FinalFinal.csv"  # Output file

# Load RDF ontology (if available) to retrieve labels
#ontology_file = "/content/envo2.owl"  # Change this to your ontology file
#g = rdflib.Graph()
#g.parse(ontology_file, format="xml")  # Load ontology

# Convert to lowercase for easier filtering
df = pd.read_csv(csv_filepath)
df = df.astype(str).applymap(lambda x: x.strip().lower())

# List of unwanted exact values
unwanted_exact_values = {"id", "comment", "type", "class", "definition", "label", "true", "envo", "database_cross_reference", "date", "chebi", "has_exact_synonym", "envoempo", "created", "creator", "editor", "	has_alternative_id", "has_related_synonym", "has_narrow_synonym", "envocmecs", "envoplastics", "alternative term", "has_broad_synonym", "term editor","creation_date", "created_by", " has_alternative_id", "has_alternative_id ", "hassynonym", "editor note", "domain", "envoatmo", "ester", "ro-eco", "curator note", "curator"}

# Regex for detecting UUID-like alphanumeric strings
uuid_pattern = re.compile(r"^[a-f0-9]{32}$|^[a-z0-9]{30,}$")

# Regex for ORCID IDs
orcid_pattern = re.compile(r"^(orcid:)?\d{4}-\d{4}-\d{4}-\d{3}[0-9xX]$")

# Function to get human-readable labels from RDF
def get_label(uri):
    if isinstance(uri, rdflib.term.URIRef):
        # Try fetching the rdfs:label for this entity
        for _, _, label_value in g.triples((uri, rdflib.RDFS.label, None)):
            return str(label_value)
        return uri.split("#")[-1] if "#" in uri else uri.split("/")[-1]
    elif isinstance(uri, rdflib.term.Literal):
        return str(uri)
    return uri  # Return as-is if it's not a URI or Literal

# Function to query ORCID API for human-readable names
def get_orcid_name(orcid_id):
    if orcid_id.startswith("orcid:"):
        orcid_id = orcid_id.split(":")[1]  # Remove "orcid:" prefix
    url = f"https://pub.orcid.org/v3.0/{orcid_id}"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        try:
            return data["person"]["name"]["given-names"]["value"] + " " + data["person"]["name"]["family-name"]["value"]
        except KeyError:
            return orcid_id  # If name not found, return original ID
    return orcid_id  # Return original if no match

# Function to retrieve labels for ontology terms (e.g., foodon:, chebi:)
def get_ontology_label(term):
    prefix_mapping = {
        "foodon": "https://www.ebi.ac.uk/ols/api/ontologies/foodon/terms?short_form=",
        "chebi": "https://www.ebi.ac.uk/ols/api/ontologies/chebi/terms?short_form="
    }

    for prefix, api_url in prefix_mapping.items():
        if term.startswith(f"{prefix}:"):
            term_id = term.split(":")[1]
            response = requests.get(api_url + term_id, headers={"Accept": "application/json"})
            if response.status_code == 200:
                data = response.json()
                if "_embedded" in data and "terms" in data["_embedded"]:
                    return data["_embedded"]["terms"][0]["label"]  # Extract label from OLS API
            return term  # If API fails, return original term

    return term  # Return original if no match

# Function to clean and replace values with human-readable labels
def clean_and_replace(value):
    if value in unwanted_exact_values:  # Exact match with unwanted words
        return None
    if value.startswith("http://") or value.startswith("https://")or value.startswith("envo:") or value.startswith("iao_") or value.startswith("foodon:") or value.startswith("contributor") or value.startswith("ro_"):  # Is a URL
        return None
    if uuid_pattern.match(value):  # Matches UUID-like strings
        return None
    #if orcid_pattern.match(value):  # Matches ORCID ID
     #   return get_orcid_name(value)
    #if ":" in value:  # Likely an ontology term
     #   return get_ontology_label(value)
    return value  # Return as-is if no match

# Apply filtering and replacement
df_cleaned = df.applymap(clean_and_replace).dropna()

# Save cleaned triples
df_cleaned.to_csv(output_csv_filepath, index=False)

print(f"\n✅ Cleaned triples saved to {output_csv_filepath} successfully!")

#import rdflib

# Load the ontology from BioPortal
#ontology_url = "https://data.bioontology.org/ontologies/OBOE-SBC/submissions/1/download?apikey=87918894-263d-4f71-9b22-bfc345c1800d"

# Initialize RDF Graph
#g = rdflib.Graph()
#g.parse(ontology_url, format="application/rdf+xml")

# Extract "is a" relations (subclasses)
#subclass_triples = []
#for s, _, o in g.triples((None, rdflib.RDFS.subClassOf, None)):
 #   subclass_triples.append((s, "is a", o))

# Extract subproperty relations (subproperties)
#subproperty_triples = []
#for s, _, o in g.triples((None, rdflib.RDFS.subPropertyOf, None)):
 #   subproperty_triples.append((s, "is a (subproperty of)", o))

# Function to format triples for readability
#def format_triples(triples):
 #   formatted = []
  #  for s, rel, o in triples:
   #     formatted.append(f"<{s}> --[{rel}]--> <{o}>")
    #return formatted

# Print "is a" subclass relations
#print("\n🔹 'Is A' (Subclass) Relations:")
#for triple in format_triples(subclass_triples):
 #   print(triple)

# Print subproperty relations
#print("\n🔹 'Is A' (Subproperty) Relations:")
#for triple in format_triples(subproperty_triples):
 #   print(triple)

!pip install pronto

import pronto
import csv

# Define ontology file path
obo_file_path = "/content/drive/MyDrive/AquaDiva/LLMTriples/chebi.obo"  # Update this with your actual file path

# Load the ChEBI ontology
ontology = pronto.Ontology(obo_file_path)

# List to store extracted triples
triples = []

# Iterate through all terms in the ontology
for term in ontology.terms():
    subject = term.name  # Get the human-readable name of the term
    subject_id = term.id  # ChEBI ID of the term
    if not subject:
        continue  # Skip terms without names

    # Extract "is_a" (subclass) relationships
    for parent in term.superclasses(distance=1):  # Direct parents only
        triples.append((subject, "is a", parent.id))  # Using ChEBI ID as object

    # Extract "relationship" (custom properties)
    for relation, targets in term.relationships.items():  # Fix: Use `.relationships`
        for target in targets:
            triples.append((subject, relation.name, target.id))  # Relation ID for clarity

# Save extracted triples to a CSV file
csv_filepath = "/content/drive/MyDrive/AquaDiva/BioPortalTriples/chebi_triples.csv"
with open(csv_filepath, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Subject", "Predicate", "Object"])  # CSV header
    writer.writerows(triples)

# Print extracted triples
print("\n🔹 Extracted ChEBI Triples:")
for triple in triples[:10]:  # Print only first 10 triples for preview
    print(f"({triple[0]}) -[{triple[1]}]-> ({triple[2]})")

print(f"\n✅ ChEBI triples saved to {csv_filepath} successfully!")

import pronto
import pandas as pd

# Load the ChEBI ontology
obo_file_path = "/content/drive/MyDrive/AquaDiva/LLMTriples/chebi.obo"  # Update this with your actual file path
ontology = pronto.Ontology(obo_file_path)

# Load the extracted triples CSV
csv_filepath = "/content/drive/MyDrive/AquaDiva/BioPortalTriples/chebi_triples.csv"  # Input CSV with IDs
output_csv_filepath = "/content/drive/MyDrive/AquaDiva/BioPortalTriples/chebi_triples_named.csv"  # Output CSV with names

# Load the CSV into a pandas DataFrame
df = pd.read_csv(csv_filepath)

# Function to get the name of a ChEBI ID
def get_chebi_name(chebi_id):
    if isinstance(chebi_id, str) and chebi_id.startswith("CHEBI:"):
        term = ontology.get(chebi_id)  # Look up the term in the ontology
        return term.name if term else chebi_id  # Return the name if found, otherwise keep the ID
    return chebi_id  # If not a ChEBI ID, return as is

# Apply the function to replace ChEBI IDs in the "Object" column
df["Object"] = df["Object"].apply(get_chebi_name)

# Save the updated triples to a new CSV file
df.to_csv(output_csv_filepath, index=False)

# Print a preview of the updated triples
print("\n🔹 Updated ChEBI Triples (with Names):")
print(df.head(10))  # Show the first 10 rows

print(f"\n✅ Named ChEBI triples saved to {output_csv_filepath} successfully!")

from rdflib import Graph

def convert_ontology_to_rdf(ontology_file, output_file):
    # Create a Graph
    g = Graph()

    # Parse the ontology file
    g.parse(ontology_file, format='xml')  # Assuming OWL/XML format

    # Serialize the graph into RDF triples (Turtle format)
    rdf_triples = g.serialize(format='turtle')

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(rdf_triples)

    print(f"RDF triples have been saved to {output_file}")

# Example usage
ontology_file = "/content/envo2.owl"  # Path to the ontology file
output_file = "/content/test_envo2.owl"  # Output RDF file in Turtle format
convert_ontology_to_rdf(ontology_file, output_file)

import pandas as pd

# Load the named triples CSV
input_csv_filepath = "/content/drive/MyDrive/AquaDiva/BioPortalTriples/chebi_triples_named.csv"  # Input file
output_csv_filepath = "/content/drive/MyDrive/AquaDiva/BioPortalTriples/chebi_triples_cleaned.csv"  # Output file

# Load CSV into pandas DataFrame
df = pd.read_csv(input_csv_filepath)

# Function to check if any cell in the row has more than 25 characters
def filter_short_entries(row):
    return all(len(str(cell)) <= 25 for cell in row)

# Apply the filter
df_cleaned = df[df.apply(filter_short_entries, axis=1)]

# Save the cleaned triples to a new CSV file
df_cleaned.to_csv(output_csv_filepath, index=False)

# Print a preview of the cleaned triples
print("\n🔹 Cleaned ChEBI Triples (≤ 25 chars per cell):")
print(df_cleaned.head(10))  # Show the first 10 rows

print(f"\n✅ Cleaned triples saved to {output_csv_filepath} successfully!")

import rdflib
import re
import csv
from rdflib.namespace import OWL, RDF, RDFS

# Path to the downloaded OWL file (update this if needed)
local_owl_file = "/content/oboe-sbc.owl"  # Change this to your actual file path
csv_output_file = "/content/oboe-sbc.csv"  # Output file path

# Initialize RDF Graph
g = rdflib.Graph()

# Load the ontology from the local OWL file
g.parse(local_owl_file, format="xml")  # OWL files are typically XML-based

# Function to retrieve the human-readable label for an ontology term
def get_label(uri):
    """ Returns the rdfs:label (name) of a term if available, otherwise its local name. """
    if isinstance(uri, rdflib.BNode):  # Ignore blank nodes
        return None
    if isinstance(uri, rdflib.URIRef):
        for _, _, label in g.triples((uri, RDFS.label, None)):  # Check for rdfs:label
            return str(label)
        return uri.split("#")[-1] if "#" in uri else uri.split("/")[-1]  # Fallback: Use last part of URI
    elif isinstance(uri, rdflib.Literal):
        return str(uri)  # If it's already readable, return as is
    return uri  # Return as-is if not a URI or Literal

# Regex to detect random alphanumeric strings (UUID-like)
uuid_pattern = re.compile(r"^[a-f0-9]{30,}$")

# Function to check if a value is invalid (UUID-like or blank node)
def is_invalid(value):
    return value is None or uuid_pattern.match(value)  # Returns True if value is None or a UUID

# List to store extracted triples
triples = []

### 🔹 Step 1: Extract Standard RDF Triples (Filtering Out "subClassOf" and Random Strings) ###
all_properties = set(g.predicates())

for prop in all_properties:
    if prop in {RDFS.subClassOf, OWL.equivalentClass}:  # Skip generic relations
        continue
    for s, _, o in g.triples((None, prop, None)):
        subject, predicate, object_ = get_label(s), get_label(prop), get_label(o)

        # Ignore triples with UUID-like strings or blank nodes
        if is_invalid(subject) or is_invalid(predicate) or is_invalid(object_):
            continue

        triples.append([subject, predicate, object_])

### 🔹 Step 2: Extract OWL Property Restrictions (Complex Relations) ###
for restriction in g.subjects(RDF.type, OWL.Restriction):
    for _, _, prop in g.triples((restriction, OWL.onProperty, None)):
        predicate = get_label(prop)

        for s, _, _ in g.triples((None, None, restriction)):
            subject = get_label(s)

            for _, _, obj in g.triples((restriction, OWL.someValuesFrom, None)):
                object_ = get_label(obj)

                # Ignore triples with UUID-like strings or blank nodes
                if is_invalid(subject) or is_invalid(predicate) or is_invalid(object_):
                    continue

                triples.append([subject, predicate, object_])

# Save extracted triples to a CSV file
with open(csv_output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Subject", "Predicate", "Object"])  # CSV header
    writer.writerows(triples)

# Print confirmation
print(f"\n✅ {len(triples)} meaningful triples extracted and saved to {csv_output_file} successfully!")

import pandas as pd

# Define file paths
file1 = "/content/envo_triples.csv"  # First CSV file
file2 = "/content/drive/MyDrive/AquaDiva/BioPortalTriples/ENVO_cleaned.csv"  # Second CSV file
output_file = "/content/drive/MyDrive/AquaDiva/BioPortalTriples/ENVO_cleaned_Final.csv"  # Output CSV file

# Load the two CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Combine the files (stack them together)
df_combined = pd.concat([df1, df2], ignore_index=True)

# Remove duplicate rows (Optional: Remove this line if you want all rows)
df_combined.drop_duplicates(inplace=True)

# Save the combined file
df_combined.to_csv(output_file, index=False)

print(f"\n✅ Combined CSV saved to: {output_file}")

