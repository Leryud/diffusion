import matplotlib.pyplot as plt
import torch
from src.unet import SinusoidalPositionEmbeddings

# Create an instance of the class
dim = 512
pe = SinusoidalPositionEmbeddings(dim)

# Generate embeddings for a range of positions
max_len = 100
positions = torch.arange(max_len).float()
embeddings = pe(positions)

# Plot the embeddings
plt.figure(figsize=(12, 6))
plt.title("Sinusoidal Positional Embeddings")
plt.xlabel("Position")
plt.ylabel("Embedding Value")
plt.matshow(embeddings.detach().numpy())
plt.show()


# def check_embeddings(embeddings):
#     # Check shape
#     assert embeddings.shape == (
#         max_len,
#         dim,
#     ), f"Expected shape ({max_len}, {dim}), got {embeddings.shape}"

#     # Check if the first half is sine and second half is cosine
#     half_dim = dim // 2
#     assert torch.allclose(
#         embeddings[:, :half_dim], torch.sin(embeddings[:, :half_dim])
#     ), "First half should be sine"
#     assert torch.allclose(
#         embeddings[:, half_dim:], torch.cos(embeddings[:, half_dim:])
#     ), "Second half should be cosine"

#     # Check if the frequencies decrease
#     freq = torch.abs(embeddings[1:] - embeddings[:-1])
#     assert torch.all(freq[1:] <= freq[:-1]), "Frequencies should decrease"

#     # Check if embeddings for different positions are unique
#     assert (
#         len(torch.unique(embeddings, dim=0)) == max_len
#     ), "Embeddings for different positions should be unique"

#     print("All checks passed!")


# check_embeddings(embeddings)


# Test the embeddings
pe = SinusoidalPositionEmbeddings(dim=64)
test_time = torch.tensor([1, 100])
result = pe(test_time)

# Plot the embeddings
plt.figure(figsize=(12, 6))
plt.plot(result.detach().numpy().T)
plt.title("Sinusoidal Positional Embeddings")
plt.xlabel("Dimension")
plt.ylabel("Embedding Value")
plt.legend([f"Time={t}" for t in test_time])
plt.show()
