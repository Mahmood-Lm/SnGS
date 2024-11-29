import torch
import open_clip
import copy

# Load the pre-trained (zero-shot) model
model_zeroshot, _, preprocess_train = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai"
)

# Initialize the fine-tuned model
model_finetuned, _, _ = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained=None
)

try:
    # Load the fine-tuned model
    checkpoint = torch.load("/home/Mahmood/soccernet/sn-gamestate/pretrained_models/jersey/V3_epoch_5.pt", map_location="cuda")

    if 'state_dict' in checkpoint:
        model_finetuned.load_state_dict(checkpoint['state_dict'])
    else:
        model_finetuned.load_state_dict(checkpoint)

    # Print parameters after loading checkpoint
    # print("Parameters after loading checkpoint:", list(model_finetuned.parameters())[0][0][:5])
except Exception as e:
    print(f"Error loading fine-tuned model: {e}")

# Print weights of the zero-shot model
print("Zero-shot model weights:", list(model_zeroshot.parameters())[0][0][:5])

# Print weights of the fine-tuned model
print("Fine-tuned model weights:", list(model_finetuned.parameters())[0][0][:5])

# Create a new model instance (a deep copy of the fine-tuned model)
model_wiseft = copy.deepcopy(model_finetuned)

# Get state dictionaries (weights) of the pre-trained and fine-tuned models
theta_0 = model_zeroshot.state_dict()
theta_1 = model_finetuned.state_dict()

# Ensure both models have the same layers
assert set(theta_0.keys()) == set(theta_1.keys())

# Set the mixing coefficient (alpha)
alpha = 0.5  # You can adjust this value

# Interpolate the weights
theta = {
    key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
    for key in theta_0.keys()
}

# Load the combined weights into the new model instance
model_wiseft.load_state_dict(theta)

# Print weights of the WiSE-FT model
print("WiSE-FT model weights:", list(model_wiseft.parameters())[0][0][:5])

# Save the WiSE-FT model
torch.save(model_wiseft.state_dict(), "/home/Mahmood/soccernet/sn-gamestate/wise-ft/wft_models/wft_model_0.5.pt")