import torch
import torch.nn as nn
import torch.optim as optim

import math


class LinearFF(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearFF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.act = nn.ReLU()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))

        self.reset_parameters()

        self.opt = optim.Adam(self.parameters(), lr=0.03)

        self.threshold = 2.0

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        # this is basically the direction of the input,
        # x_hat
        input_norm = input / (input.norm(p=2, dim=1, keepdim=True) + 1e-4)

        # matmul with weights, forward pass
        input_mat = torch.mm(input_norm, self.weight.T)

        input_relu = self.act(input_mat)

        return input_relu

    def goodness(self, input):
        return self(input).pow(2).mean(dim=1)

    def train(self, x_pos, x_neg, num_epochs):
        for epoch in range(num_epochs):

            g_pos = self.goodness(x_pos)
            g_neg = self.goodness(x_neg)

            # we want to increase goodness for g_pos
            # so we negate it and add our threshold
            # the optimizer will try to reduce -g_pos
            # which will increase g_pos or goodness of pos input
            loss_pos = torch.log(1 + torch.exp(-g_pos + self.threshold))

            # we want to decrease goodness for g_neg
            # so we subtract the threshold
            # optimizer will try to reduce g_neg
            # which will decrease goodness of neg input
            loss_neg = torch.log(1 + torch.exp(g_neg - self.threshold))

            # finally our loss is sum of both
            loss = loss_pos.mean() + loss_neg.mean()

            self.opt.zero_grad()

            loss.backward()

            self.opt.step()

        # done with epochs, return outputs
        # detach because we dont want to gradients to flow back
        return self(x_pos).detach(), self(x_neg).detach()


class FFMLP(nn.Module):
    def __init__(self, device):
        super(FFMLP, self).__init__()

        self.layers = [
            LinearFF(in_features=784, out_features=300).to(device),
            LinearFF(in_features=300, out_features=300).to(device),
            LinearFF(in_features=300, out_features=300).to(device),
        ]

    def train(self, x_pos, x_neg, num_epochs):
        x_pos_hat, x_neg_hat = x_pos, x_neg
        for layer in self.layers:
            x_pos_hat, x_neg_hat = layer.train(x_pos_hat, x_neg_hat, num_epochs)
