import os
import random
import json
from collections import defaultdict

def save_statistics(data, path):
    """Function to save statistics about the dataset to a file."""
    entity_type_counts = defaultdict(int)
    relation_type_counts = defaultdict(int)

    for entry in data:
        for entity in entry["entity_list"]:
            entity_type_counts[entity["type"][0]] += 1
        for relation in entry["relation_list"]:
            relation_type_counts[relation["predicate"]] += 1

    with open(path, "w") as file:
        file.write("Total Entries: {}\n".format(len(data)))
        file.write("Total Entities: {}\n".format(sum([len(entry["entity_list"]) for entry in data])))
        file.write("Total Relations: {}\n".format(sum([len(entry["relation_list"]) for entry in data])))
        file.write("Unique Entity Types: {}\n".format(len(entity_type_counts)))
        file.write("Unique Relation Types: {}\n".format(len(relation_type_counts)))
        file.write("\nEntity Type Counts:\n")
        for ent_type, count in entity_type_counts.items():
            file.write(f"{ent_type}: {count}\n")
        file.write("\nRelation Type Counts:\n")
        for rel_type, count in relation_type_counts.items():
            file.write(f"{rel_type}: {count}\n")

# Load the converted data

data_paths = [os.path.join('utils/datasets/datasets_tokens',i)  for i in os.listdir('utils/datasets/datasets_tokens')]
data = []
for dataset_path in data_paths:
    with open(dataset_path, "r") as file:
        data = data + (json.load(file))



# Extract entity and relation types for the ent2id.json and rel2id.json
entity_types = set()
relation_types = set()
for entry in data:
    for entity in entry["entity_list"]:
        entity_types.add(entity["type"][0])
    for relation in entry["relation_list"]:
        relation_types.add(relation["predicate"])

# Convert sets to dictionaries with unique IDs
ent2id = {ent: idx for idx, ent in enumerate(sorted(entity_types))}
rel2id = {rel: idx for idx, rel in enumerate(sorted(relation_types))}

# Save the ent2id.json and rel2id.json
with open("datasets/my_data/ent2id.json", "w") as file:
    json.dump(ent2id, file)
with open("datasets/my_data/rel2id.json", "w") as file:
    json.dump(rel2id, file)

# Create and save data_statistics.txt
save_statistics(data, "datasets/my_data/data_statistics.txt")
