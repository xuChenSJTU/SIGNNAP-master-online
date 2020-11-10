import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearClassifier(nn.Module):
    def __init__(self, input_dim=128, output_dim=7, dropout=0.5):
        super(LinearClassifier, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.fc = nn.Linear(self.input_dim, self.output_dim)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, feats):
        x = self.fc(feats)
        x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)