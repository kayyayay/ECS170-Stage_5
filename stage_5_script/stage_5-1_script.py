# stage_5_1_dataset_inspection.py

import os
import torch

from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader

# Find project root automatically
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

print("Project Root:")
print(PROJECT_ROOT)

datasets = ["cora", "citeseer", "pubmed"]

for dataset_name in datasets:

    print("\n" + "=" * 60)
    print(f"Dataset: {dataset_name}")
    print("=" * 60)

    # Create loader
    loader = Dataset_Loader(
        dName=dataset_name,
        dDescription=f"{dataset_name} dataset"
    )

    # Set dataset path
    loader.dataset_source_folder_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "stage_5_data",
        dataset_name
    )

    print("\nDataset Path:")
    print(loader.dataset_source_folder_path)

    # Debugging checks
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

    # Load dataset
    data = loader.load()

    graph = data["graph"]

    X = graph["X"]
    y = graph["y"]
    edges = graph["edge"]

    print("\nDataset Statistics")
    print("-" * 40)

    print(f"Number of nodes: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of edges: {edges.shape[0]}")
    print(f"Number of classes: {len(torch.unique(y))}")

    print("\nFirst 5 Node Labels:")
    print(y[:5])

    print("\nFirst 5 Edges:")
    print(edges[:5])

    print("\nFirst Node Feature Vector (first 20 features):")
    print(X[0][:20])

    print("\nClass Distribution:")

    unique_labels, counts = torch.unique(
        y,
        return_counts=True
    )

    for label, count in zip(unique_labels, counts):
        print(
            f"Class {label.item()}: {count.item()} nodes"
        )

print("\nDataset inspection completed successfully.")