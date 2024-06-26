import torch

# Configuration parameters
IMG_SIZE = 64
BATCH_SIZE = 128
T = 300
BETA_START = 0.0001
BETA_END = 0.02

train_samples = "src/results/train"

# Check for CUDA GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
# Check for MPS (Metal Performance Shaders) availability
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
# Fallback to CPU
else:
    device = torch.device("cpu")
    print("Using CPU")
    if not torch.backends.mps.is_built():
        print(
            "MPS not available because the current PyTorch install was not built with MPS enabled."
        )
    else:
        print(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )


lr = 0.001
epochs = 100
