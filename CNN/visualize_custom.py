import os
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

# Import the AdvancedCNN architecture from the training script
# Assuming the file `train_cnn.py` defines AdvancedCNN in the same directory
from train_cnn import AdvancedCNN, DEFAULT_MODEL_PATH

# Device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Image transforms for inference (no augmentation)
val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the trained model
model = AdvancedCNN().to(device)
model.load_state_dict(torch.load(DEFAULT_MODEL_PATH, map_location=device))
model.eval()

# Path to custom images folder (relative to this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
custom_dir = os.path.join(script_dir, 'customImage')

# Gather image file paths (jpg, jpeg, png)
image_files = [f for f in os.listdir(custom_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

predictions = []
for fname in image_files:
    img_path = os.path.join(custom_dir, fname)
    # Load and preprocess image, skip if unreadable
    try:
        pil_img = Image.open(img_path).convert('RGB')
    except (UnidentifiedImageError, OSError) as e:
        print(f"Skipping unreadable image {fname}: {e}")
        continue
    img_tensor = val_transforms(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor).squeeze()
        prob = torch.sigmoid(logits).item()
        label = 'DOG' if prob > 0.5 else 'CAT'
        confidence = prob if label == 'DOG' else 1 - prob
    predictions.append((pil_img, label, confidence, fname))

# Visualize: create a grid of the images with titles
cols = 3
rows = (len(predictions) + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
axs = axs.flatten() if isinstance(axs, (list, tuple, np.ndarray)) else [axs]

for ax, (pil_img, label, conf, fname) in zip(axs, predictions):
    ax.imshow(pil_img)
    ax.set_title(f"{fname}\n{label} ({conf*100:.1f}% )", fontsize=10)
    ax.axis('off')

# Hide any unused subplots
for i in range(len(predictions), len(axs)):
    axs[i].axis('off')

plt.tight_layout()
output_path = os.path.join(script_dir, 'custom_predictions.png')
plt.savefig(output_path)
print(f"Visualization saved to {output_path}")
