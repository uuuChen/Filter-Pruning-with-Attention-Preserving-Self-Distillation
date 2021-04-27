import torch.nn as nn
from pruner import FiltersPruningModule


class FeatureExtractor(FiltersPruningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._features = dict()

        for name, child in self.model.named_modules():
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                child.register_forward_hook(self.save_outputs_hook(name))

    def save_outputs_hook(self, layer_name):
        def fn(_, __, output):
            self._features[layer_name] = output
        return fn

    def forward(self, x):
        out = self.model(x)
        return out, self._features
