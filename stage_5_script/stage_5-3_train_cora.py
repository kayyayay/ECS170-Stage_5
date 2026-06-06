import os

from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from code.stage_5_code.Method_GCN_Node_Classification import Method_GCN_Node_Classification
from code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy

# get the project root directory
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
# folder for saving results
RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "result",
    "stage_5_result"
)
os.makedirs(RESULT_DIR, exist_ok=True) # create result directory if it does not exist

# initialize dataset loader
dataset = Dataset_Loader(
    dName='cora',
    dDescription='Cora Dataset'
)
# set path to the Cora dataset
dataset.dataset_source_folder_path = os.path.join(
    PROJECT_ROOT,
    "data",
    "stage_5_data",
    "cora"
)
# load dataset into memory
data = dataset.load()

# initialize GCN model
method = Method_GCN_Node_Classification(
    'GCN',
    'Node Classification'
)
# pass dataset to the model
method.data = data

# train the model and evaluate on the test set
result = method.run()

# create accuracy evaluator
evaluator = Evaluate_Accuracy(
    'Accuracy',
    'GCN Evaluation'
)
# provide predictions and true labels for evaluation
evaluator.data = {
    'true_y': result['true_y'],
    'pred_y': result['pred_y']
}
# compute evaluation metrics
metrics = evaluator.evaluate()
# display final performance metrics
print("\nFinal Results")
print("=" * 40)

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

import matplotlib.pyplot as plt
# Loss Curve
plt.figure(figsize=(8, 5)) # plot training loss over epochs

plt.plot(result['loss_list'])

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GCN Training Loss - Cora")

plt.grid(True)
# save loss curve figure
loss_path = os.path.join(
    RESULT_DIR,
    "cora_loss_curve.png"
)

plt.savefig(loss_path)
plt.close()
# save evaluation results to a text file
metrics_path = os.path.join(
    RESULT_DIR,
    "cora_metrics.txt"
)
# write metrics to disk
with open(metrics_path, "w") as f:

    f.write("GCN Results on Cora\n")
    f.write("=" * 30 + "\n")

    for metric, value in metrics.items():
        f.write(f"{metric}: {value:.4f}\n")
