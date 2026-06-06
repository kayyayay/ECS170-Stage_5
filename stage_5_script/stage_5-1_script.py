# stage_5_1_dataset_inspection.py

import os
import torch

from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader

# get the project root director
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

print("Project Root:")
print(PROJECT_ROOT)
# datasets to inspect
datasets = ["cora", "citeseer", "pubmed"]
# check each dataset one at a time
for dataset_name in datasets:

    print("\n" + "=" * 60)
    print(f"Dataset: {dataset_name}")
    print("=" * 60)

    # initialize dataset loader
    loader = Dataset_Loader(
        dName=dataset_name,
        dDescription=f"{dataset_name} dataset"
    )

    # build path to the dataset folder
    loader.dataset_source_folder_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "stage_5_data",
        dataset_name
    )

    print("\nDataset Path:")
    print(loader.dataset_source_folder_path)

    # verify that required files exist
    print("Dataset folder exists:",
          os.path.exists(loader.dataset_source_folder_path))

    print("Node file exists:",
          os.path.exists(
              os.path.join(loader.dataset_source_folder_path, "node")
          ))

    print("Link file exists:",
          os.path.exists(
              os.path.join(loader.dataset_source_folder_path, "link")
          ))

    # load graph data
    data = loader.load()
    # extract graph components
    graph = data["graph"]

    X = graph["X"]
    y = graph["y"]
    edges = graph["edge"]
    # display basic dataset information
    print("\nDataset Statistics")
    print("-" * 40)

    print(f"Number of nodes: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of edges: {edges.shape[0]}")
    print(f"Number of classes: {len(torch.unique(y))}")
    # show a few sample labels
    print("\nFirst 5 Node Labels:")
    print(y[:5])
    # show a few sample edges
    print("\nFirst 5 Edges:")
    print(edges[:5])
    # inspect the feature values of the first node
    print("\nFirst Node Feature Vector (first 20 features):")
    print(X[0][:20])

    print("\nClass Distribution:")
    # count how many nodes belong to each class
    unique_labels, counts = torch.unique(
        y,
        return_counts=True
    )
    # print class distribution
    for label, count in zip(unique_labels, counts):
        print(
            f"Class {label.item()}: {count.item()} nodes"
        )

print("\nDataset inspection completed successfully.")
