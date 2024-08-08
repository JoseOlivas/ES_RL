import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



np.random.seed(42)

if __name__ == "__main__":
    loaded_model = torch.load('BipedalWalkerHardcore-v3_actor.h5')