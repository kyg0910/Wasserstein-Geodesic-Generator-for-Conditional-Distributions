import numpy as np
import torch

def random_horizontal_flip(x, c):
    flipped_idx = np.random.binomial(n=1, p=0.5, size=np.shape(x)[0])
    x[flipped_idx] = torch.from_numpy(np.flip(x[flipped_idx].cpu(), axis=3).copy()).cuda()
    c[flipped_idx, 0] = -c[flipped_idx, 0]
    return x, c

# def normalize_for_alexnet(x):
#     x_repeated = x.repeat((1,3,1,1))
#     x_repeated[:, 0, :, :] = (x_repeated[:, 0, :, :] - 0.485)/0.229
#     x_repeated[:, 1, :, :] = (x_repeated[:, 1, :, :] - 0.456)/0.224
#     x_repeated[:, 2, :, :] = (x_repeated[:, 2, :, :] - 0.406)/0.225
#     return x_repeated

