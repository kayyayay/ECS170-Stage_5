'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy(evaluate):
    data = None
    average = 'macro' # 'macro' treats all classes equally, 'weighted' weights by class frequency

    def evaluate(self):
        print('evaluating performance...')

        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        results = {
            'accuracy':  accuracy_score(true_y, pred_y),
            'precision': precision_score(true_y, pred_y, average=self.average, zero_division=0),
            'recall':    recall_score(true_y, pred_y, average=self.average, zero_division=0),
            'f1':        f1_score(true_y, pred_y, average=self.average, zero_division=0),
        }

        return results