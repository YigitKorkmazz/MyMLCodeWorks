import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms

# ==========================================
# 1. SETUP & HYPERPARAMETERS
# ==========================================

# Use Apple Silicon GPU (MPS) if available, otherwise fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters (Adjust these to control training speed and quality)
BATCH_SIZE = 32          # Number of images processed at once
LEARNING_RATE = 0.001    # How fast the model learns
EPOCHS = 3               # Number of epochs per fold (reduced to 3 for faster CV training)
IMAGE_SIZE = 128         # Resize images to 128x128 pixels
SUBSET_LIMIT = 20000      # Set to None to use all 25,000 images, or a smaller number for quick testing
NUM_FOLDS = 5            # Number of cross-validation folds

# ==========================================
# 2. DATASET DEFINITION
# ==========================================

class CatDogDataset(Dataset):
    """
    A custom Dataset to load Cat and Dog images from folders.
    Inheriting from torch.utils.data.Dataset allows us to use PyTorch's DataLoader.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []  # 0 for Cat, 1 for Dog

        # We have two folders: 'Cat' and 'Dog'
        categories = ["Cat", "Dog"]
        for label, category in enumerate(categories):
            category_dir = os.path.join(root_dir, category)
            if not os.path.isdir(category_dir):
                continue
            
            print(f"Scanning images in {category} folder...")
            for filename in os.listdir(category_dir):
                # Only process image files
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(category_dir, filename)
                    
                    # Check if the file is empty (a common issue in the Kaggle Dogs vs Cats dataset)
                    if os.path.getsize(img_path) > 0:
                        self.image_paths.append(img_path)
                        self.labels.append(label)
        
        print(f"Successfully loaded {len(self.image_paths)} valid images.")

    def __len__(self):
        # Returns the total number of images
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Retrieve the image and label at the given index
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open the image using PIL (Python Imaging Library)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # If an image fails to load (corrupted file), load a dummy black image
            print(f"Error loading {img_path}: {e}. Using dummy image.")
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))

        # Apply transformations (resizing, converting to tensor, normalization)
        if self.transform:
            image = self.transform(image)

        # Convert label to float tensor (needed for BCEWithLogitsLoss later)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return image, label_tensor

# ==========================================
# 3. TRANSFORMATIONS (PREPROCESSING)
# ==========================================

# Deep learning models need all input images to be the exact same size and normalized.
data_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize to 128x128
    transforms.ToTensor(),                        # Convert PIL Image to PyTorch Tensor (scales pixels to [0.0, 1.0])
    transforms.Normalize(                         # Normalize with standard ImageNet mean and standard deviation
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# 4. DEFINE THE CNN ARCHITECTURE
# ==========================================

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN).
    It extracts features using convolutional layers and classifies them using fully connected layers.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Conv Block 1: Input (3 channels, 128x128) -> Output (16 channels, 128x128)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # MaxPooling downsamples image dimensions by 2: (16 channels, 128x128) -> (16 channels, 64x64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2: Input (16 channels, 64x64) -> Output (32 channels, 64x64)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # Pool2: (32 channels, 64x64) -> (32 channels, 32x32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 3: Input (32 channels, 32x32) -> Output (64 channels, 32x32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        # Pool3: (64 channels, 32x32) -> (64 channels, 16x16)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flattened size: 64 channels * 16 pixels * 16 pixels = 16,384 features
        # Fully Connected (FC) Layers
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Prevents overfitting by randomly turning off 50% of neurons during training
        self.fc2 = nn.Linear(128, 1)    # Outputs 1 single value (logit) representing probability score of Dog vs Cat

    def forward(self, x):
        # Forward pass: defining how data flows through the network layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        # Flatten the multi-dimensional feature maps into a 1D vector for the linear layers
        x = x.view(x.size(0), -1)

        # Linear classification layers
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # We output raw numbers (logits). PyTorch's BCEWithLogitsLoss handles sigmoid internally for stability.
        return x

# ==========================================
# 5. TRAINING & VALIDATION LOOPS
# ==========================================

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()  # Put the model in training mode (activates dropout)
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for images, labels in train_loader:
            # Move data to GPU (MPS) or CPU
            images, labels = images.to(device), labels.to(device)

            # 1. Clear gradients from the previous step
            optimizer.zero_grad()

            # 2. Run the forward pass to get predictions
            outputs = model(images).squeeze()  # Squeeze turns shape [batch, 1] to [batch]

            # 3. Calculate how wrong the model was (loss)
            loss = criterion(outputs, labels)

            # 4. Backward pass: Calculate gradients (derivatives of loss with respect to weights)
            loss.backward()

            # 5. Update weights based on gradients
            optimizer.step()

            # Track statistics
            running_loss += loss.item() * images.size(0)
            
            # Since outputs are logits, a logit > 0 means model predicts Dog (label 1), <= 0 means Cat (label 0)
            predictions = (outputs > 0).float()
            correct_preds += (predictions == labels).sum().item()
            total_preds += labels.size(0)

        epoch_train_loss = running_loss / total_preds
        epoch_train_acc = (correct_preds / total_preds) * 100

        # --- VALIDATION PHASE ---
        model.eval()  # Put the model in evaluation mode (deactivates dropout)
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # Turn off gradient calculation to save memory and speed up
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

    print("--- Training Complete! ---")
    # Return model weights along with the final epoch metrics
    return model, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc

# ==========================================
# 6. MAIN EXECUTION PIPELINE
# ==========================================

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to your PetImages directory containing Cat/ and Dog/ folders
    dataset_path = os.path.join(script_dir, "PetImages")

    # Initialize dataset
    full_dataset = CatDogDataset(root_dir=dataset_path, transform=data_transforms)

    # Optional: limit dataset size to train faster (for practice/debugging)
    if SUBSET_LIMIT and SUBSET_LIMIT < len(full_dataset):
        print(f"Limiting dataset to {SUBSET_LIMIT} images for faster execution...")
        # Randomly select a subset
        indices = torch.randperm(len(full_dataset))[:SUBSET_LIMIT]
        dataset_subset = torch.utils.data.Subset(full_dataset, indices)
    else:
        dataset_subset = full_dataset

    # For cross-validation, we split the indices into NUM_FOLDS segments.
    # Set seed for reproducible fold splits
    torch.manual_seed(42)
    dataset_size = len(dataset_subset)
    indices = torch.randperm(dataset_size).tolist()
    fold_size = dataset_size // NUM_FOLDS

    # Lists to store final train/val performance for each fold
    fold_train_losses = []
    fold_train_accs = []
    fold_val_losses = []
    fold_val_accs = []

    print(f"\n--- Starting {NUM_FOLDS}-Fold Cross Validation ---")
    print(f"Total dataset subset size: {dataset_size} images")
    print(f"Each fold uses {fold_size} validation images and {dataset_size - fold_size} training images.")

    for fold in range(NUM_FOLDS):
        print(f"\n==========================================")
        print(f"           FOLD {fold + 1} / {NUM_FOLDS}")
        print(f"==========================================")

        # Slice the validation and training indices
        val_indices = indices[fold * fold_size : (fold + 1) * fold_size]
        train_indices = indices[:fold * fold_size] + indices[(fold + 1) * fold_size:]

        # Create Subset datasets
        train_dataset = torch.utils.data.Subset(dataset_subset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset_subset, val_indices)

        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 1. Initialize a clean CNN model (so we train from scratch on each fold)
        model = SimpleCNN().to(device)

        # 2. Define Loss and Optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 3. Train the model on this fold's training set, validating on the fold's validation set
        trained_model, train_loss, train_acc, val_loss, val_acc = train_model(
            model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS
        )

        # Record metrics from the final epoch of this fold
        fold_train_losses.append(train_loss)
        fold_train_accs.append(train_acc)
        fold_val_losses.append(val_loss)
        fold_val_accs.append(val_acc)

        # Save weights from the final fold so the inference function has a model to load
        if fold == NUM_FOLDS - 1:
            save_path = os.path.join(script_dir, "cat_dog_cnn.pth")
            torch.save(trained_model.state_dict(), save_path)
            print(f"\nSaved final fold model weights to {save_path}")

    # Calculate average performance across all folds
    avg_train_loss = sum(fold_train_losses) / NUM_FOLDS
    avg_train_acc = sum(fold_train_accs) / NUM_FOLDS
    avg_val_loss = sum(fold_val_losses) / NUM_FOLDS
    avg_val_acc = sum(fold_val_accs) / NUM_FOLDS

    # Print final summary of cross-validation results
    print("\n" + "="*50)
    print("      CROSS VALIDATION SUMMARY METRICS")
    print("="*50)
    for f in range(NUM_FOLDS):
        print(f"Fold {f+1} | Train Loss: {fold_train_losses[f]:.4f} | Train Acc: {fold_train_accs[f]:.2f}% | "
              f"Val (Test) Loss: {fold_val_losses[f]:.4f} | Val (Test) Acc: {fold_val_accs[f]:.2f}%")
    print("-"*50)
    print(f"AVERAGE | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | "
          f"Val (Test) Loss: {avg_val_loss:.4f} | Val (Test) Acc: {avg_val_acc:.2f}%")
    print("="*50)

# ==========================================
# 7. INFERENCE HELPER FUNCTION
# ==========================================

# Default model path in the same directory as the script
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cat_dog_cnn.pth")

def predict_single_image(image_path, model_weights_path=DEFAULT_MODEL_PATH):
    """
    Utility function to classify a single new image using the trained model weights.
    """
    # 1. Load the model architecture
    model = SimpleCNN()
    
    # 2. Load the trained weights
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    model.eval()  # Put in evaluation mode
    
    # 3. Load and preprocess the target image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Could not open image: {e}")
        return
        
    image_tensor = data_transforms(image).unsqueeze(0)  # unsqueeze adds a batch dimension [1, 3, 128, 128]
    
    # 4. Perform prediction (no gradients needed)
    with torch.no_grad():
        output = model(image_tensor).squeeze()  # Get raw logit output
        
        # Apply Sigmoid to convert logit to a probability between 0.0 and 1.0
        probability = torch.sigmoid(output).item()
        
        # Label 0 = Cat, Label 1 = Dog
        if probability > 0.5:
            print(f"Prediction: DOG ({probability * 100:.2f}% confidence)")
        else:
            print(f"Prediction: CAT ({(1 - probability) * 100:.2f}% confidence)")
