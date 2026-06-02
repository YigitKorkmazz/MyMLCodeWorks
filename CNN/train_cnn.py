import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# ==========================================
# 1. SETUP & HYPERPARAMETERS
# ==========================================

# Use Apple Silicon GPU (MPS) if available, otherwise fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 32          # Number of images processed at once
LEARNING_RATE = 0.001    # How fast the model learns
EPOCHS = 10              # Number of epochs per fold (increase for better accuracy)
IMAGE_SIZE = 256         # Upgraded from 128 to 256 - captures finer details
SUBSET_LIMIT = 10000      # Set to None to use all 25,000 images
NUM_FOLDS = 5            # Number of cross-validation folds

# ==========================================
# 2. TRANSFORMATIONS (PREPROCESSING)
# ==========================================

# TRAINING transforms: We apply data augmentation here.
# The model will see a slightly different version of each image every epoch,
# which prevents it from just memorizing the training data (overfitting).
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),        # Resize to 256x256
    transforms.RandomHorizontalFlip(),                  # Flip image left-right 50% of the time
    transforms.RandomRotation(degrees=15),              # Rotate up to +/-15 degrees randomly
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Randomly tweak brightness and contrast
    transforms.ToTensor(),                              # Convert PIL Image to PyTorch Tensor
    transforms.Normalize(                               # Normalize to ImageNet statistics
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# VALIDATION transforms: NO augmentation here.
# We want to evaluate the model on clean, unmodified images.
val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# 3. DATASET DEFINITION
# ==========================================

class CatDogDataset(Dataset):
    """
    A custom Dataset that loads Cat and Dog images from folders.
    NOTE: This version stores raw PIL images without any transforms applied.
    The transform is applied later via the TransformSubset wrapper (see below),
    which allows us to apply different augmentations for training vs validation.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []  # 0 for Cat, 1 for Dog

        categories = ["Cat", "Dog"]
        for label, category in enumerate(categories):
            category_dir = os.path.join(root_dir, category)
            if not os.path.isdir(category_dir):
                print(f"Warning: Directory not found: {category_dir}")
                continue

            print(f"Scanning images in {category} folder...")
            for filename in os.listdir(category_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(category_dir, filename)
                    # Skip empty (corrupted) files
                    if os.path.getsize(img_path) > 0:
                        self.image_paths.append(img_path)
                        self.labels.append(label)

        print(f"Successfully loaded {len(self.image_paths)} valid image paths.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Return the raw PIL image and label (transform applied by TransformSubset)
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}. Using dummy image.")
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return image, label_tensor


class TransformSubset(Dataset):
    """
    A wrapper that applies a specific transform to a subset of a dataset.
    This is the key tool that lets us use different augmentations on
    the training split vs the validation split of the same base dataset.
    """
    def __init__(self, subset, transform):
        self.subset = subset        # The underlying dataset subset
        self.transform = transform  # The transform to apply

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Get the raw image and label from the subset
        image, label = self.subset[idx]
        # Apply our transform (train_transforms or val_transforms) to the raw PIL image
        image = self.transform(image)
        return image, label


# ==========================================
# 4. ADVANCED CNN MODEL ARCHITECTURE
# ==========================================

class AdvancedCNN(nn.Module):
    """
    A 4-layer Convolutional Neural Network with Batch Normalization.
    This mirrors the Keras architecture with 32->64->128->256 filters.
    
    Key improvements over SimpleCNN:
    - 4 Conv layers instead of 3 (deeper = more complex features)
    - Batch Normalization after every layer (stabilizes and speeds up training)
    - Two dense (FC) layers instead of one (more powerful classifier)
    - 256x256 input instead of 128x128 (finer image detail)
    """
    def __init__(self):
        super(AdvancedCNN, self).__init__()

        # --- FEATURE EXTRACTION ---
        # With padding='valid' (padding=0), each Conv2d reduces spatial dim by 2:
        # e.g., 256 -> conv -> 254 -> pool -> 127

        # Conv Block 1: 3 channels -> 32 filters
        # Size: (3, 256, 256) -> conv -> (32, 254, 254) -> pool -> (32, 127, 127)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(32)   # Normalizes the 32 output channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2: 32 -> 64 filters
        # Size: (32, 127, 127) -> conv -> (64, 125, 125) -> pool -> (64, 62, 62)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 3: 64 -> 128 filters
        # Size: (64, 62, 62) -> conv -> (128, 60, 60) -> pool -> (128, 30, 30)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 4: 128 -> 256 filters
        # Size: (128, 30, 30) -> conv -> (256, 28, 28) -> pool -> (256, 14, 14)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # After 4 conv+pool blocks: 256 channels * 14 * 14 = 50,176 features
        # --- CLASSIFIER ---

        # Dense Layer 1: 50176 -> 128
        self.fc1 = nn.Linear(256 * 14 * 14, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)   # BatchNorm1d for dense (1D) layers
        self.dropout1 = nn.Dropout(0.5)

        # Dense Layer 2: 128 -> 64
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.5)

        # Output Layer: 64 -> 1 (single logit for binary classification)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # Pass input through all 4 Conv Blocks
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))

        # Flatten all feature maps into a 1D vector
        x = x.view(x.size(0), -1)  # Shape: [batch_size, 50176]

        # Pass through Classifier layers
        x = self.dropout1(torch.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)  # Output raw logit (BCEWithLogitsLoss applies sigmoid internally)

        return x


# ==========================================
# 5. TRAINING & VALIDATION LOOPS
# ==========================================

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()  # Activates BatchNorm and Dropout
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 1. Clear previous gradients
            optimizer.zero_grad()

            # 2. Forward pass
            outputs = model(images).squeeze()  # Shape: [batch_size]

            # 3. Calculate loss
            loss = criterion(outputs, labels)

            # 4. Backward pass: compute gradients
            loss.backward()

            # 5. Update model weights
            optimizer.step()
            # Step the LR scheduler after each epoch (outside the batch loop – will be called later)
            running_loss += loss.item() * images.size(0)
            predictions = (outputs > 0).float()  # logit > 0 means Dog (label=1)
            correct_preds += (predictions == labels).sum().item()
            total_preds += labels.size(0)

        epoch_train_loss = running_loss / total_preds
        epoch_train_acc = (correct_preds / total_preds) * 100

        # --- VALIDATION PHASE ---
        model.eval()  # Deactivates BatchNorm (uses running stats) and Dropout
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # Disable gradient calculation (saves memory)
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                predictions = (outputs > 0).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        # Update learning rate schedule at the end of each epoch
        scheduler.step()

    print("--- Training Complete! ---")
    # Return model and final-epoch metrics for the cross-validation summary
    return model, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc


# ==========================================
# 6. MAIN EXECUTION PIPELINE (CROSS VALIDATION)
# ==========================================

if __name__ == "__main__":
    # Resolve paths relative to the script file so it works from any directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "PetImages")

    # Load the base dataset (without any transforms - raw PIL images)
    full_dataset = CatDogDataset(root_dir=dataset_path)

    # Optionally limit the dataset to a subset for quicker experimentation
    if SUBSET_LIMIT and SUBSET_LIMIT < len(full_dataset):
        print(f"Limiting dataset to {SUBSET_LIMIT} images for faster execution...")
        indices = torch.randperm(len(full_dataset))[:SUBSET_LIMIT]
        dataset_subset = torch.utils.data.Subset(full_dataset, indices)
    else:
        dataset_subset = full_dataset

    # --- 5-FOLD CROSS VALIDATION ---
    torch.manual_seed(42)
    dataset_size = len(dataset_subset)
    indices = torch.randperm(dataset_size).tolist()
    fold_size = dataset_size // NUM_FOLDS

    fold_train_losses, fold_train_accs = [], []
    fold_val_losses, fold_val_accs = [], []

    print(f"\n--- Starting {NUM_FOLDS}-Fold Cross Validation ---")
    print(f"Total dataset: {dataset_size} images | "
          f"Each fold: {dataset_size - fold_size} train / {fold_size} val")

    for fold in range(NUM_FOLDS):
        print(f"\n==========================================")
        print(f"           FOLD {fold + 1} / {NUM_FOLDS}")
        print(f"==========================================")

        # Split indices into train and validation for this fold
        val_indices   = indices[fold * fold_size : (fold + 1) * fold_size]
        train_indices = indices[:fold * fold_size] + indices[(fold + 1) * fold_size:]

        # Create raw subsets (no transforms yet)
        raw_train_subset = torch.utils.data.Subset(dataset_subset, train_indices)
        raw_val_subset   = torch.utils.data.Subset(dataset_subset, val_indices)

        # Wrap each subset with the appropriate transform:
        # Training gets augmentation; Validation gets only resize + normalize
        train_dataset = TransformSubset(raw_train_subset, train_transforms)
        val_dataset   = TransformSubset(raw_val_subset, val_transforms)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize a FRESH model for every fold
        model = AdvancedCNN().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        # Cosine Annealing LR scheduler – slowly reduces LR over epochs for smoother convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        # Train and validate on this fold
        trained_model, train_loss, train_acc, val_loss, val_acc = train_model(
            model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS
        )

        fold_train_losses.append(train_loss)
        fold_train_accs.append(train_acc)
        fold_val_losses.append(val_loss)
        fold_val_accs.append(val_acc)

        # Save the weights from the last fold for use in inference
        if fold == NUM_FOLDS - 1:
            save_path = os.path.join(script_dir, "cat_dog_cnn.pth")
            torch.save(trained_model.state_dict(), save_path)
            print(f"\nSaved final fold model weights to {save_path}")

    # --- PRINT FINAL SUMMARY ---
    avg_train_loss = sum(fold_train_losses) / NUM_FOLDS
    avg_train_acc  = sum(fold_train_accs)   / NUM_FOLDS
    avg_val_loss   = sum(fold_val_losses)   / NUM_FOLDS
    avg_val_acc    = sum(fold_val_accs)     / NUM_FOLDS

    print("\n" + "="*58)
    print("           CROSS VALIDATION SUMMARY METRICS")
    print("="*58)
    for f in range(NUM_FOLDS):
        print(f"Fold {f+1} | Train Loss: {fold_train_losses[f]:.4f} | "
              f"Train Acc: {fold_train_accs[f]:.2f}% | "
              f"Val (Test) Loss: {fold_val_losses[f]:.4f} | "
              f"Val (Test) Acc: {fold_val_accs[f]:.2f}%")
    print("-"*58)
    print(f"AVERAGE | Train Loss: {avg_train_loss:.4f} | "
          f"Train Acc: {avg_train_acc:.2f}% | "
          f"Val (Test) Loss: {avg_val_loss:.4f} | "
          f"Val (Test) Acc: {avg_val_acc:.2f}%")
    print("="*58)


# ==========================================
# 7. INFERENCE HELPER FUNCTION
# ==========================================

# Default model path (same directory as this script)
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cat_dog_cnn.pth")

def predict_single_image(image_path, model_weights_path=DEFAULT_MODEL_PATH):
    """
    Classifies a single image as Cat or Dog using the trained model.
    """
    # Load the model architecture and weights
    model = AdvancedCNN()
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    model.eval()  # Deactivates Dropout for inference

    # Load and preprocess the image (use val_transforms - no augmentation)
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Could not open image: {e}")
        return

    image_tensor = val_transforms(image).unsqueeze(0)  # Add batch dimension: [1, 3, 256, 256]

    with torch.no_grad():
        output = model(image_tensor).squeeze()
        probability = torch.sigmoid(output).item()  # Convert logit to probability (0-1)

    if probability > 0.5:
        print(f"Prediction: DOG ({probability * 100:.2f}% confidence)")
    else:
        print(f"Prediction: CAT ({(1 - probability) * 100:.2f}% confidence)")
