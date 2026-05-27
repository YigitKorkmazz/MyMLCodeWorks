import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set premium styling for matplotlib
plt.rcParams['figure.facecolor'] = '#121214'
plt.rcParams['axes.facecolor'] = '#1a1a1e'
plt.rcParams['text.color'] = '#e2e8f0'
plt.rcParams['axes.labelcolor'] = '#e2e8f0'
plt.rcParams['xtick.color'] = '#94a3b8'
plt.rcParams['ytick.color'] = '#94a3b8'
plt.rcParams['grid.color'] = '#334155'
plt.rcParams['grid.alpha'] = 0.4
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

# 1. Load the dataset using pandas
print("Loading housing.csv...")
df = pd.read_csv("housing.csv")

# Clean index column
if "" in df.columns:
    df = df.drop(columns=[""])
elif df.columns[0] == "Unnamed: 0":
    df = df.drop(columns=[df.columns[0]])

print(f"Dataset shape: {df.shape} (506 rows, 14 columns)")
print("Columns: ", list(df.columns))

# -------------------------------------------------------------
# Part 1: Single Variable Linear Regression (Predict medv using rm)
# -------------------------------------------------------------
print("\n=== Experiment 1: Single Variable Regression (Rooms vs. Price) ===")

# torch.tensor(data, dtype=torch.float32): Converts NumPy array of values into a float32 PyTorch Tensor.
# .unsqueeze(1): Changes the dimension of the tensor from shape (506) to (506, 1).
# PyTorch linear layers require inputs to be shaped as (batch_size, num_features).
X_single_raw = torch.tensor(df["rm"].values, dtype=torch.float32).unsqueeze(1)
y_single = torch.tensor(df["medv"].values, dtype=torch.float32).unsqueeze(1)

# Feature Scaling (Standardization: Z-score normalization)
# - X_single_raw.mean(): Computes the mean value of the entire tensor.
# - X_single_raw.std(): Computes the standard deviation of the tensor.
X_single_mean = X_single_raw.mean()
X_single_std = X_single_raw.std()
X_single = (X_single_raw - X_single_mean) / X_single_std

# Define Model by inheriting from nn.Module
class SingleRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear(1, 1): Creates a linear layer with 1 input (scaled rooms) and 1 output (predicted price).
        self.linear = nn.Linear(1,1)
        
    def forward(self, x):
        return self.linear(x)

single_model = SingleRegressionModel()
criterion = nn.MSELoss()  # Mean Squared Error Loss
single_optimizer = optim.SGD(single_model.parameters(), lr=0.05)  # SGD Optimizer

single_loss_history = []
epochs = 150

for epoch in range(1, epochs + 1):
    single_model.train()
    # Passing input tensor to the model automatically calls the forward() method.
    predictions = single_model(X_single)
    
    # Calculate loss between predictions and true labels
    loss = criterion(predictions, y_single)
    
    # Zero gradients, calculate gradients (backpropagation), and perform updates
    single_optimizer.zero_grad()
    loss.backward()
    single_optimizer.step()
    
    single_loss_history.append(loss.item())
    
    if epoch % 25 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{epochs} | Loss (MSE): {loss.item():.4f}")

# Extract final single variable parameters
# .item() extracts the singular float value out of the weight/bias tensors.
w_single = single_model.linear.weight.item()
b_single = single_model.linear.bias.item()
print(f"Trained model: Price = {w_single:.4f} * Rooms_scaled + {b_single:.4f}")

# -------------------------------------------------------------
# Part 2: Multi-Variable Linear Regression (Predict medv using all 13 features)
# -------------------------------------------------------------
print("\n=== Experiment 2: Multi-Variable Regression (All Features vs. Price) ===")

feature_cols = [col for col in df.columns if col != "medv"]

# X_multi_raw: We extract all 13 columns. The shape is (506, 13).
X_multi_raw = torch.tensor(df[feature_cols].values, dtype=torch.float32)
y_multi = torch.tensor(df["medv"].values, dtype=torch.float32).unsqueeze(1)

# Multi-Feature Scaling
# - X_multi_raw.mean(dim=0): Calculates the mean of each feature column individually (along dimension 0).
#   The result is a tensor of shape (13), containing the mean of each of the 13 columns.
# - X_multi_raw.std(dim=0): Calculates the standard deviation of each feature column.
X_multi_mean = X_multi_raw.mean(dim=0)
X_multi_std = X_multi_raw.std(dim=0)
# Prevent division by zero if std is zero
X_multi_std[X_multi_std == 0] = 1.0
X_multi = (X_multi_raw - X_multi_mean) / X_multi_std

# Define Multi-Variable Model
class MultiRegressionModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # We set in_features=num_features (13 inputs, since we use all 13 features).
        self.linear = nn.Linear(in_features=num_features, out_features=1)
        
    def forward(self, x):
        return self.linear(x)

# Instantiate the model with 13 features
multi_model = MultiRegressionModel(num_features=len(feature_cols))
multi_optimizer = optim.SGD(multi_model.parameters(), lr=0.05)

multi_loss_history = []

for epoch in range(1, epochs + 1):
    multi_model.train()
    predictions = multi_model(X_multi)
    loss = criterion(predictions, y_multi)
    
    multi_optimizer.zero_grad()
    loss.backward()
    multi_optimizer.step()
    
    multi_loss_history.append(loss.item())
    
    if epoch % 25 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{epochs} | Loss (MSE): {loss.item():.4f}")

# Analyze Multi-Variable Coefficients
# - multi_model.linear.weight.detach(): Tells PyTorch to create a copy of the weights tensor 
#   that is disconnected from the computational graph (so updates are no longer tracked).
# - .numpy(): Converts the PyTorch tensor to a standard NumPy array.
# - .flatten(): Flattens the array from shape (1, 13) to (13) for easy reading.
weights = multi_model.linear.weight.detach().numpy().flatten()
bias = multi_model.linear.bias.item()

print("\n--- Multi-Variable Model Coefficients (Impact of normalized features) ---")
coefficients = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': weights
})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
for idx, row in coefficients.iterrows():
    print(f"  {row['Feature']:8s}: {row['Coefficient']:7.4f}")
print(f"  Intercept (bias): {bias:.4f}")

# -------------------------------------------------------------
# Part 3: Visualizations
# -------------------------------------------------------------
print("\nGenerating housing_regression_results.png...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Linear Regression on Boston Housing Dataset in PyTorch", 
             fontsize=16, fontweight='bold', color='#f8fafc', y=1.02)

# Subplot 1: Single variable scatter and regression line
ax1 = axes[0]
# X_single_raw.numpy() and y_single.numpy(): Converts tensors to NumPy arrays for Matplotlib plotting.
ax1.scatter(X_single_raw.numpy(), y_single.numpy(), color='#38bdf8', alpha=0.5, edgecolors='none', s=25, label='Actual Data')

# Compute regression line
rooms_range = np.linspace(df["rm"].min(), df["rm"].max(), 100)
# Convert rooms range to PyTorch tensor
rooms_range_tensor = torch.tensor(rooms_range, dtype=torch.float32).unsqueeze(1)
# Scale using the training statistics
rooms_range_scaled = (rooms_range_tensor - X_single_mean) / X_single_std
# torch.no_grad(): Disables gradient tracking during predictions (inference)
with torch.no_grad():
    predictions_range = single_model(rooms_range_scaled).numpy()

ax1.plot(rooms_range, predictions_range, color='#f43f5e', linewidth=3, label=f'Fit: Price = {w_single:.2f}*rm_scaled + {b_single:.2f}')
ax1.set_title("Single Variable Fit (Rooms vs. Price)", fontsize=13, fontweight='semibold', color='#f1f5f9', pad=10)
ax1.set_xlabel("Average Number of Rooms per Dwelling (rm)", fontsize=11)
ax1.set_ylabel("Median House Value ($1000s) (medv)", fontsize=11)
ax1.grid(True)
ax1.legend(facecolor='#1e1e24', edgecolor='#334155')

# Subplot 2: Loss Convergence Comparison
ax2 = axes[1]
ax2.plot(single_loss_history, label='Single Feature Model (rm)', color='#fb7185', linewidth=2.5)
ax2.plot(multi_loss_history, label='Multi-Feature Model (13 variables)', color='#34d399', linewidth=2.5)
ax2.set_title("Training Loss Convergence Comparison", fontsize=13, fontweight='semibold', color='#f1f5f9', pad=10)
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("Mean Squared Error (MSE) Loss", fontsize=11)
ax2.grid(True)
ax2.legend(facecolor='#1e1e24', edgecolor='#334155')

plt.tight_layout()
plt.savefig("housing_regression_results.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()
print("Plots generated successfully!")
