import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BaseGenerator(nn.Module):
    def __init__(self, generator, opt):
        super(BaseGenerator, self).__init__()
        self.generator = generator
        self.opt = opt

    def forward(self, inputs):
        return self.generator(inputs.contiguous().view(-1, inputs.size(-1)))

    def backward(self, outputs, targets, weights, normalizer, criterion, regression=False):
        logits = outputs.contiguous().view(-1) if regression else self.forward(outputs)

        loss = criterion(logits, targets.contiguous().view(-1), weights.contiguous().view(-1))
        loss.div(normalizer).backward()
        loss = loss.data[0]
        return loss

    def predict(self, outputs, targets, weights, criterion):
        logits = self.forward(outputs)
        preds = logits.data.max(1)[1].view(outputs.size(0), -1)

        loss = criterion(logits, targets.contiguous().view(-1), weights.contiguous().view(-1)).data[0]

        return preds, loss

