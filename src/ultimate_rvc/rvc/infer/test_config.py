import torch

checkpoint_path = 'argos.pth'
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model_config = checkpoint["config"]
weights = checkpoint["weight"]

labels_correct_order = [
    "hidden_channels",
    "filter_channels",
    "inter_channels",
    "n_heads",
    "n_layers",
    "kernel_size",
    "spk_embed_dim",
    "upsample_initial_channel",
    "upsample_rates",
    "upsample_kernel_sizes",
    "gin_channels",
    "sampling_rate"
]

print("\nModel Configuration (Corrected):")
print("===============================")
for idx, label in enumerate(labels_correct_order):
    value = model_config[idx]
    print(f"{label:30}: {value}")

# Confirm speaker embeddings from weights explicitly:
spk_embed_dim_weights = checkpoint["weight"]["emb_g.weight"].shape[0]
print(f"\nDynamic spk_embed_dim from weights: {spk_embed_dim_weights}")

# Verify consistency explicitly:
print("\nConsistency Check:")
print("==============================")
print(f"Sampling rate from config     : {model_config[-1]}")
print(f"Sampling rate from checkpoint : {checkpoint.get('sr', 'Not provided')}")
print(f"Use F0 (pitch)                : {checkpoint.get('f0', 1)}")
print(f"Version                       : {checkpoint.get('version', 'Unknown')}")
