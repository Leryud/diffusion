import torch

# Configuration parameters
IMG_SIZE = 64
BATCH_SIZE = 128
T = 300
BETA_START = 0.0001
BETA_END = 0.02

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print(
            "MPS not available because the current PyTorch install was not "
            "built with MPS enabled."
        )
        device = torch.device("cpu")
    else:
        print(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )
        device = torch.device("cpu")

else:
    device = torch.device("mps")

lr = 0.001
epochs = 100
