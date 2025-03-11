

!curl -H "Authorization: apikey token=BIOPortalAPIToken" "https://data.bioontology.org/ontologies/CHEBI/classes?page=1"

!curl "https://data.bioontology.org/ontologies/CHEBI/classes?page=1&apikey=87918894-263d-4f71-9b22-bfc345c1800d"

response = requests.get(BASE_URL, params={"page": 1, "apikey": "87918894-263d-4f71-9b22-bfc345c1800d"})

import requests
import json

API_KEY = "87918894-263d-4f71-9b22-bfc345c1800d"  # Replace with your actual API key
BASE_URL = "https://data.bioontology.org/ontologies/OBOE-SBC/classes"

def get_all_classes():
    """Fetches all classes (name + description) from the OBOE-SBC ontology using BioPortal API."""
    classes_dict = {}  # Dictionary to store class names and descriptions
    page = 1

    while True:
        response = requests.get(BASE_URL, params={"page": page, "apikey": API_KEY})

        if response.status_code != 200:
            print(f"❌ Error {response.status_code}: {response.text}")
            break

        data = response.json()

        if "collection" in data:
            ontology_classes = data["collection"]
        else:
            print("⚠️ Unexpected response structure. Check API response.")
            print(data)
            break

        if not ontology_classes:
            print("✅ No more data. Retrieval complete.")
            break

        # Extract class labels and descriptions
        for cls in ontology_classes:
            label = cls.get("prefLabel", "Unnamed")
            description = cls.get("definition", [""])
            if isinstance(description, list):  # Sometimes, definitions are lists
                description = description[0] if description else ""

            classes_dict[label] = description  # Store in dictionary

        print(f"📄 Fetched page {page}, total classes so far: {len(classes_dict)}")

        page += 1  # Move to the next page

    return classes_dict

# Run the function and get all classes
oboe_sbc_classes = get_all_classes()

# Print summary
print(f"🎉 Retrieved {len(oboe_sbc_classes)} classes from OBOE-SBC!")

# Save the data to a JSON file
json_filename = "oboe_sbc_classes.json"
with open(json_filename, "w", encoding="utf-8") as f:
    json.dump(oboe_sbc_classes, f, indent=2, ensure_ascii=False)

print(f"💾 Classes saved to '{json_filename}'")

import requests
import json

API_KEY = "87918894-263d-4f71-9b22-bfc345c1800d"  # Replace with your actual API key
BASE_URL = "https://data.bioontology.org/ontologies/ENVO/classes"

def get_all_classes():
    """Fetches all classes (name + description) from the ENVO ontology using BioPortal API."""
    classes_dict = {}  # Dictionary to store class names and descriptions
    page = 1

    while True:
        response = requests.get(BASE_URL, params={"page": page, "apikey": API_KEY})

        if response.status_code != 200:
            print(f"❌ Error {response.status_code}: {response.text}")
            break

        data = response.json()

        if "collection" in data:
            ontology_classes = data["collection"]
        else:
            print("⚠️ Unexpected response structure. Check API response.")
            print(data)
            break

        if not ontology_classes:
            print("✅ No more data. Retrieval complete.")
            break

        # Extract class labels and descriptions
        for cls in ontology_classes:
            label = cls.get("prefLabel", "Unnamed")
            description = cls.get("definition", [""])
            if isinstance(description, list):  # Sometimes, definitions are lists
                description = description[0] if description else ""

            classes_dict[label] = description  # Store in dictionary

        print(f"📄 Fetched page {page}, total classes so far: {len(classes_dict)}")

        page += 1  # Move to the next page

    return classes_dict

# Run the function and get all classes
envo_classes = get_all_classes()

# Print summary
print(f"🎉 Retrieved {len(envo_classes)} classes from ENVO!")

# Save the data to a JSON file
json_filename = "envo_classes.json"
with open(json_filename, "w", encoding="utf-8") as f:
    json.dump(envo_classes, f, indent=2, ensure_ascii=False)

print(f"💾 Classes saved to '{json_filename}'")

import requests
import json

API_KEY = "87918894-263d-4f71-9b22-bfc345c1800d"  # Replace with your actual API key
BASE_URL = "https://data.bioontology.org/ontologies/CHEBI/classes"

def get_all_classes():
    """Fetches all classes (name + description) from the ChEBI ontology using BioPortal API."""
    classes_dict = {}  # Dictionary to store class names and descriptions
    page = 1

    while True:
        response = requests.get(BASE_URL, params={"page": page, "apikey": API_KEY})

        if response.status_code != 200:
            print(f"❌ Error {response.status_code}: {response.text}")
            break

        data = response.json()

        if "collection" in data:
            ontology_classes = data["collection"]
        else:
            print("⚠️ Unexpected response structure. Check API response.")
            print(data)
            break

        if not ontology_classes:
            print("✅ No more data. Retrieval complete.")
            break

        # Extract class labels and descriptions
        for cls in ontology_classes:
            label = cls.get("prefLabel", "Unnamed")
            description = cls.get("definition", [""])
            if isinstance(description, list):  # Sometimes, definitions are lists
                description = description[0] if description else ""

            classes_dict[label] = description  # Store in dictionary

        print(f"📄 Fetched page {page}, total classes so far: {len(classes_dict)}")

        page += 1  # Move to the next page

    return classes_dict

# Run the function and get all classes
chebi_classes = get_all_classes()

# Print summary
print(f"🎉 Retrieved {len(chebi_classes)} classes from ChEBI!")

# Save the data to a JSON file
json_filename = "chebi_classes.json"
with open(json_filename, "w", encoding="utf-8") as f:
    json.dump(chebi_classes, f, indent=2, ensure_ascii=False)

print(f"💾 Classes saved to '{json_filename}'")

