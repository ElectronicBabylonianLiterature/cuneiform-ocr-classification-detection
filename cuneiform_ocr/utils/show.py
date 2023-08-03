import matplotlib.pyplot as plt
import torch

def show(inputs):
    for i in range(inputs.shape[0]):
        show = torch.clone(inputs.cpu())[i]
        show = show.permute(1, 2, 0)
        plt.imshow(show)
        plt.show()