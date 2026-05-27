import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Load and clean the dataset
print("Loading housing.csv for Cross Validation...")
df = pd.read_csv("housing.csv")

if "" in df.columns:
    df = df.drop(columns=[""])
elif df.columns[0] == "Unnamed: 0":
    df = df.drop(columns=[df.columns[0]])

# We will use all 13 features (Multi-Variable)
feature_cols = [col for col in df.columns if col != "medv"]
X_raw = torch.tensor(df[feature_cols].values, dtype=torch.float32)
y_raw = torch.tensor(df["medv"].values, dtype=torch.float32).unsqueeze(1)

# Feature Scaling (Standardization)
X_mean = X_raw.mean(dim=0)
X_std = X_raw.std(dim=0)
X_std[X_std == 0] = 1.0
X = (X_raw - X_mean) / X_std
y = y_raw

# Define our Model architecture
class MultiRegressionModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(in_features=num_features, out_features=1)
    def forward(self, x):
        return self.linear(x)

def calculate_r_squared(y_true, y_pred):
    """
    Calculates the R-squared (R^2) score.
    R^2 represents the proportion of variance for the dependent variable that's explained by the independent variables.
    It's the closest thing to an 'accuracy' percentage in regression.
    1.0 is perfect prediction.
    """
    # Sum of squared residuals (errors)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    # Total sum of squares (variance of the data)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    
    r2 = 1 - (ss_res / ss_tot)
    return r2.item()

# -------------------------------------------------------------
# 5-Fold Cross Validation setup
# -------------------------------------------------------------
K = 5
num_samples = len(X)
epochs = 150
lr = 0.05

# Shuffle indices randomly to split the data
# torch.randperm generates a random permutation of integers from 0 to num_samples-1
indices = torch.randperm(num_samples)

fold_size = num_samples // K

print(f"\nStarting {K}-Fold Cross Validation...")
print(f"Total samples: {num_samples} | Samples per fold: ~{fold_size}\n")

test_mse_scores = []
test_r2_scores = []

for fold in range(K):
    # Calculate the start and end indices for the Test set
    start_idx = fold * fold_size
    # Ensure the last fold includes any remaining samples
    end_idx = start_idx + fold_size if fold < K - 1 else num_samples
    
    test_indices = indices[start_idx:end_idx]
    
    # Train indices are all indices EXCEPT the test_indices
    train_indices = torch.cat([indices[:start_idx], indices[end_idx:]])
    
    # Split data into Train and Test sets for this fold
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    # 1. Initialize a FRESH model and optimizer for each fold
    # (We don't want the model to remember data from the previous fold)
    model = MultiRegressionModel(num_features=len(feature_cols))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # 2. Train the model on the Training set
    model.train()
    for epoch in range(epochs):
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # 3. Evaluate the model on the unseen Test set
    # torch.no_grad() prevents tracking history, speeding up evaluation
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        test_predictions = model(X_test)
        test_loss = criterion(test_predictions, y_test)
        test_r2 = calculate_r_squared(y_test, test_predictions)
        
    test_mse_scores.append(test_loss.item())
    test_r2_scores.append(test_r2)
    
    print(f"Fold {fold+1}/{K} | Test MSE: {test_loss.item():7.4f} | Test R^2 (Accuracy): {test_r2 * 100:6.2f}%")

# -------------------------------------------------------------
# Final Results
# -------------------------------------------------------------
avg_mse = sum(test_mse_scores) / K
avg_r2 = sum(test_r2_scores) / K

print("\n" + "="*40)
print(f"CROSS VALIDATION RESULTS (Average over {K} folds)")
print("="*40)
print(f"Average Test MSE Loss  : {avg_mse:.4f}")
print(f"Average Test R^2 Score : {avg_r2 * 100:.2f}%")
print("="*40)
print("Interpretation: The model can explain approximately {:.2f}% of the variance in house prices using unseen data!".format(avg_r2 * 100))
