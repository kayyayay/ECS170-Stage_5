import torch
import torch.nn as nn
import torch.nn.functional as F

from code.base_class.method import method

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features)
        )

        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):

        support = torch.mm(x, self.weight)

        output = torch.spmm(adj, support)

        return output

class Method_GCN_Node_Classification(method, nn.Module):
    def __init__(self,
                 mName=None,
                 mDescription=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.learning_rate = 0.01
        self.max_epoch = 200
        self.hidden_dim = 16
        self.dropout = 0.5

        self.loss_list = []
        self.acc_list = []

    def build_model(self, nfeat, nclass):
        self.gc1 = GraphConvolution(
            nfeat,
            self.hidden_dim
        )

        self.gc2 = GraphConvolution(
            self.hidden_dim,
            nclass
        )

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=5e-4
        )

    def forward(self, x, adj):
        x = F.relu(
            self.gc1(x, adj)
        )

        x = F.dropout(
            x,
            self.dropout,
            training=self.training
        )

        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=1)

    def train_model(
            self,
            features,
            labels,
            adj,
            idx_train):

        for epoch in range(self.max_epoch):

            self.train()

            self.optimizer.zero_grad()

            output = self.forward(
                features,
                adj
            )

            loss = F.nll_loss(
                output[idx_train],
                labels[idx_train]
            )

            loss.backward()

            self.optimizer.step()

            self.loss_list.append(loss.item())

            pred = output[idx_train].max(1)[1]

            acc = pred.eq(
                labels[idx_train]
            ).float().mean().item()

            self.acc_list.append(acc)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch} "
                    f"Loss {loss.item():.4f} "
                    f"Acc {acc:.4f}"
                )

    def test(
            self,
            features,
            labels,
            adj,
            idx_test):

        self.eval()

        with torch.no_grad():
            output = self.forward(
                features,
                adj
            )

            pred = output[idx_test].max(1)[1]

        return {
            'pred_y': pred.cpu().numpy(),
            'true_y': labels[idx_test].cpu().numpy()
        }

    def run(self):
        graph = self.data['graph']

        features = graph['X']
        labels = graph['y']
        adj = graph['utility']['A']

        split = self.data['train_test_val']

        idx_train = split['idx_train']
        idx_test = split['idx_test']

        self.build_model(
            features.shape[1],
            len(torch.unique(labels))
        )

        self.train_model(
            features,
            labels,
            adj,
            idx_train
        )

        result = self.test(
            features,
            labels,
            adj,
            idx_test
        )

        result['loss_list'] = self.loss_list
        result['acc_list'] = self.acc_list

        return result