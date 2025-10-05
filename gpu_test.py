import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(
    f"GPU name: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'None'}")
print(
    f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'None'}")
