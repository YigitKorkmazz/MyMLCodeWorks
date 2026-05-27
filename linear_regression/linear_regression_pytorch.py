import torch
# torch.nn (Neural Networks): PyTorch's library of building blocks for neural networks (e.g. layers, activation functions, loss functions).
import torch.nn as nn
# torch.optim (Optimizers): PyTorch's library of optimization algorithms like SGD, Adam, RMSprop.
import torch.optim as optim

# Set random seeds for reproducibility
torch.manual_seed(42)

# 1. Define the model by subclassing nn.Module
# nn.Module is the base class for all neural network modules in PyTorch. 
# Your custom models must inherit from it.
class LinearRegressionModel(nn.Module):
    def __init__(self):
        # super().__init__() is required to correctly initialize the parent nn.Module class.
        super().__init__()
        
        # nn.Linear(in_features, out_features): A pre-defined PyTorch layer representing y = x * W^T + bias.
        # - in_features=1: We pass 1 input (X).
        # - out_features=1: We want 1 output prediction (y_pred).
        # PyTorch will automatically create weight (W) and bias tensors for this layer.
        self.linear = nn.Linear(in_features=1, out_features=1)
        
    def forward(self, x):
        # The forward method defines the forward pass (computational flow) of the model.
        # It is triggered automatically when you call "model(x)".
        return self.linear(x)

def generate_synthetic_data(w_true, b_true, num_samples=100):
    """
    Generates synthetic linear data with some Gaussian noise:
    y = w * X + b + noise
    """
    # PyTorch's linear layers expect inputs of shape (batch_size, num_features).
    # Since we have 100 samples and 1 feature, we shape X as (100, 1) using rand(num_samples, 1).
    X = (torch.rand(num_samples, 1) * 4 - 2)
    noise = torch.randn(num_samples, 1) * 0.4
    y = w_true * X + b_true + noise
    return X, y

def train_pytorch_way(X, y, epochs=100, lr=0.1):
    print("\n--- Training the PyTorch Way (nn.Module & torch.optim) ---")
    
    # Instantiate our model object
    model = LinearRegressionModel()
    
    # We can inspect parameters directly using .weight and .bias attributes of our linear layer.
    initial_w = model.linear.weight.item()
    initial_b = model.linear.bias.item()
    print(f"Initial parameters: w = {initial_w:.4f}, b = {initial_b:.4f}")
    
    # nn.MSELoss(): Instantiates Mean Squared Error loss.
    # This acts as our mathematical criterion for how "wrong" the model's predictions are.
    criterion = nn.MSELoss()
    
    # optim.SGD(parameters, lr): Instantiates the Stochastic Gradient Descent optimizer.
    # - model.parameters(): Tells the optimizer which weights/biases to update (in this case, self.linear's W and bias).
    # - lr: The learning rate.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        # model.train(): Puts the model in "training mode".
        # While not strictly necessary for simple linear layers, it is a crucial habit because 
        # certain layers (like Dropout and BatchNorm) behave differently during training vs. evaluation.
        model.train()
        
        # A. Forward pass: compute predictions.
        # Passing X directly to model(X) triggers the forward() method under the hood.
        predictions = model(X)
        
        # B. Compute loss by comparing predictions to true y values
        loss = criterion(predictions, y)
        
        # C. Zero the gradients.
        # optimizer.zero_grad() clears the gradients of all parameters being optimized.
        # This is equivalent to manual w.grad.zero_() and b.grad.zero_() in our scratch script.
        optimizer.zero_grad()
        
        # D. Backward pass: compute gradients.
        # Calculates derivatives of loss with respect to weights and bias.
        loss.backward()
        
        # E. Update weights.
        # optimizer.step() updates parameters using the calculated gradients: param = param - lr * param.grad
        # This replaces our manual update step from scratch!
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            current_w = model.linear.weight.item()
            current_b = model.linear.bias.item()
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.6f} | w: {current_w:.4f} | b: {current_b:.4f}")
            
    final_w = model.linear.weight.item()
    final_b = model.linear.bias.item()
    return final_w, final_b

if __name__ == "__main__":
    TRUE_W = 2.5
    TRUE_B = 1.0
    
    print(f"Generating synthetic data with true w = {TRUE_W} and true b = {TRUE_B}...")
    X, y = generate_synthetic_data(TRUE_W, TRUE_B, num_samples=100)
    
    # Train the model
    w_fit, b_fit = train_pytorch_way(X, y, epochs=100, lr=0.1)
    
    print("\n=== Final Comparison ===")
    print(f"True Values     : w = {TRUE_W:.4f}, b = {TRUE_B:.4f}")
    print(f"PyTorch trained : w = {w_fit:.4f}, b = {b_fit:.4f}")
