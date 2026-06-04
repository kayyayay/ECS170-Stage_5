import os

from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from code.stage_5_code.Method_GCN_Node_Classification import Method_GCN_Node_Classification
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy

# Project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "result",
    "stage_5_result"
)
os.makedirs(RESULT_DIR, exist_ok=True)

# Load Cora dataset
dataset = Dataset_Loader(
    dName='pubmed',
    dDescription='Pubmed Dataset'
)

dataset.dataset_source_folder_path = os.path.join(
    PROJECT_ROOT,
    "data",
    "stage_5_data",
    "pubmed"
)

data = dataset.load()

# Create GCN model
method = Method_GCN_Node_Classification(
    'GCN',
    'Node Classification'
)

method.data = data

# Train + Test
result = method.run()

# Evaluate
evaluator = Evaluate_Accuracy(
    'Accuracy',
    'GCN Evaluation'
)

evaluator.data = {
    'true_y': result['true_y'],
    'pred_y': result['pred_y']
}

metrics = evaluator.evaluate()

print("\nFinal Results")
print("=" * 40)

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

import matplotlib.pyplot as plt
# Loss Curve
plt.figure(figsize=(8, 5))

plt.plot(result['loss_list'])

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GCN Training Loss - Pubmed")

plt.grid(True)

loss_path = os.path.join(
    RESULT_DIR,
    "pubmed_loss_curve.png"
)

plt.savefig(loss_path)
plt.close()

metrics_path = os.path.join(
    RESULT_DIR,
    "pubmed_metrics.txt"
)

with open(metrics_path, "w") as f:

    f.write("GCN Results on Pubmed/n")
    f.write("=" * 30 + "\n")

    for metric, value in metrics.items():
        f.write(f"{metric}: {value:.4f}\n")