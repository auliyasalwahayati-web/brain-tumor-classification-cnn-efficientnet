import torch
import torchvision

print("PyTorch :", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("CUDA :", torch.cuda.is_available())

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)