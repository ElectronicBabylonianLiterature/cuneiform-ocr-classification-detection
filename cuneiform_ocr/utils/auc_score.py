import torch
from torcheval.metrics.aggregation.auc import AUC
metric = AUC()
metric.update(
    torch.tensor([0, 0.7378,  .7362, .7351, .7295, .7317, .7030, 1]),
    torch.tensor([0, .7797, .7829, .785, .7931, .7887, .8106, 1] )
)
print(metric.compute())