import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from code.base_class.method import method

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        # learnable weight matrix
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features)
        )
        # learnable bias term
        self.bias = nn.Parameter(
            torch.FloatTensor(out_features)
        )
        # initialize model parameters
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # apply linear transformation
        support = torch.mm(x, self.weight)
        # aggregate information from neighbors
        output = torch.spmm(adj, support)
        return output + self.bias


class Method_GCN_Node_Classification(method, nn.Module):

    def __init__(self, mName=None, mDescription=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # training settings
        self.learning_rate = 0.01
        self.max_epoch = 400
        self.hidden_dim = 64
        self.dropout = 0.5
        self.patience = 50
        # store training history
        self.loss_list = []
        self.acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []

    def build_model(self, nfeat, nclass):
        # first graph convolution layer
        self.gc1 = GraphConvolution(nfeat, self.hidden_dim)
        # output graph convolution layer
        self.gc2 = GraphConvolution(self.hidden_dim, nclass)
        # apply weight decay only to weights
        params_with_decay = [
            p for name, p in self.named_parameters() if 'bias' not in name
        ]
        # do not apply weight decay to bias terms
        params_no_decay = [
            p for name, p in self.named_parameters() if 'bias' in name
        ]

        self.optimizer = torch.optim.Adam([
            {'params': params_with_decay, 'weight_decay': 5e-4},
            {'params': params_no_decay,   'weight_decay': 0.0}
        ], lr=self.learning_rate)

    def forward(self, x, adj):
        # first GCN layer with ReLU activation
        x = F.relu(self.gc1(x, adj))
        # dropout for regularization
        x = F.dropout(x, self.dropout, training=self.training)
        # output layer
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def train_model(self, features, labels, adj, idx_train, idx_val):
        # keep track of the best validation loss
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.max_epoch):

            # --- Training step ---
            self.train()
            self.optimizer.zero_grad()

            output = self.forward(features, adj) # forward pass on training data

            loss = F.nll_loss(output[idx_train], labels[idx_train]) # compute training loss
            # update model parameters
            loss.backward()
            self.optimizer.step()

            self.loss_list.append(loss.item())
            # calculate training accuracy
            pred = output[idx_train].max(1)[1]
            acc = pred.eq(labels[idx_train]).float().mean().item()
            self.acc_list.append(acc)

            # evaluate model on validation set
            self.eval()
            with torch.no_grad():
                val_output = self.forward(features, adj)
                val_loss = F.nll_loss(
                    val_output[idx_val], labels[idx_val]
                ).item()
                val_pred = val_output[idx_val].max(1)[1]
                val_acc = val_pred.eq(
                    labels[idx_val]
                ).float().mean().item()
            # save metrics for plotting later
            self.val_loss_list.append(val_loss)
            self.val_acc_list.append(val_acc)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch} | "
                    f"Loss {loss.item():.4f} Acc {acc:.4f} | "
                    f"Val Loss {val_loss:.4f} Val Acc {val_acc:.4f}"
                )

            # save model if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1 # stop training if validation loss does not improve
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        # Restore the best weights found during training
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

    def test(self, features, labels, adj, idx_test):
        self.eval() # switch to evaluation mode
        with torch.no_grad():
            output = self.forward(features, adj)
            pred = output[idx_test].max(1)[1] # get predicted class labels

        return {
            'pred_y': pred.cpu().numpy(),
            'true_y': labels[idx_test].cpu().numpy()
        }

    def run(self):
        graph = self.data['graph'] # load graph data

        features = graph['X']
        labels = graph['y']
        adj = graph['utility']['A']

        split = self.data['train_test_val'] # get train, validation, and test splits
        idx_train = split['idx_train']
        idx_test = split['idx_test']
        idx_val = split['idx_val']

        # create GCN model
        self.build_model(
            features.shape[1],
            len(torch.unique(labels))
        )

        # train the model
        self.train_model(features, labels, adj, idx_train, idx_val)
        # evaluate on the test set
        result = self.test(features, labels, adj, idx_test)

        result['loss_list'] = self.loss_list
        result['acc_list'] = self.acc_list
        result['val_loss_list'] = self.val_loss_list
        result['val_acc_list'] = self.val_acc_list

        return result
