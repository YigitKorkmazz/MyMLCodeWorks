import torch
import random

# torch.manual_seed(42): Sets the seed for PyTorch's random number generator.
# This ensures that random initializations (like rand and randn) produce the exact same numbers every time we run the script.
torch.manual_seed(42)
random.seed(42)

def generate_synthetic_data(w_true, b_true, num_samples=100):
    """
    Generates synthetic linear data with some Gaussian noise:
    y = w * X + b + noise
    """
    # torch.rand(num_samples): Generates a tensor of shape (num_samples) containing random numbers 
    # uniformly distributed between 0 and 1.
    # We multiply by 4 and subtract 2 to scale it so X is randomly spread between -2 and 2.
    X = torch.rand(num_samples) * 4 - 2
    
    # torch.randn(num_samples): Generates random numbers from a Standard Normal (Gaussian) distribution 
    # (mean = 0, standard deviation = 1).
    # We multiply by 0.4 to scale the noise so it is smaller.
    noise = torch.randn(num_samples) * 0.4
    
    # Perform element-wise tensor math: multiply X by weight, add bias, add noise.
    y = w_true * X + b_true + noise
    return X, y

def train_batch_gradient_descent(X, y, epochs=100, lr=0.1):
    """
    Trains linear regression using Batch Gradient Descent (GD) from scratch.
    Calculates gradient on the entire dataset at each step.
    """
    print("\n--- Training with Batch Gradient Descent (GD) ---")
    
    # torch.randn(1): Generates a single random number from a standard normal distribution.
    # requires_grad=True: Tells PyTorch's Autograd engine to track every math operation performed on this tensor.
    # This allows PyTorch to automatically calculate derivatives (gradients) with respect to this tensor later.
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    
    # w.item() and b.item(): Converts a PyTorch tensor containing exactly 1 value into a regular Python float.
    # (Useful for formatting print statements).
    print(f"Initial parameters: w = {w.item():.4f}, b = {b.item():.4f}")
    
    for epoch in range(1, epochs + 1):
        # 2. Forward Pass: compute prediction y_pred
        y_pred = w * X + b
        
        # torch.mean(tensor): Calculates the average of all elements in a tensor.
        # Here we calculate the Mean Squared Error (MSE).
        loss = torch.mean((y_pred - y) ** 2)
        
        # loss.backward(): Triggers PyTorch's backpropagation.
        # It calculates the derivative of the 'loss' tensor with respect to every tracked parameter (w and b).
        # The resulting gradients are stored inside w.grad and b.grad.
        loss.backward()
        
        # torch.no_grad(): A context manager that disables gradient tracking.
        # We use it here because we are manually updating our weights (w = w - lr * w.grad).
        # We don't want PyTorch to track these update calculations in its computational graph!
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            
            # w.grad.zero_() and b.grad.zero_():
            # In PyTorch, calling .backward() accumulates (adds) gradients to the .grad attribute.
            # It does NOT overwrite them. If we don't manually reset them to 0 after updating,
            # the gradients will grow larger and larger every epoch.
            w.grad.zero_()
            b.grad.zero_()
            
        if epoch % 10 == 0 or epoch == 1:
            # loss.item(): Extracts the single loss value as a standard Python number.
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.6f} | w: {w.item():.4f} | b: {b.item():.4f}")
            
    return w.item(), b.item()

def train_stochastic_gradient_descent(X, y, epochs=10, lr=0.01):
    """
    Trains linear regression using Stochastic Gradient Descent (SGD) from scratch.
    Calculates gradient and updates parameters using a single sample at a time.
    """
    print("\n--- Training with Stochastic Gradient Descent (SGD) ---")
    
    # Initialize weights randomly, telling PyTorch to track their gradients.
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    
    num_samples = len(X)
    print(f"Initial parameters: w = {w.item():.4f}, b = {b.item():.4f}")
    
    for epoch in range(1, epochs + 1):
        indices = list(range(num_samples))
        random.shuffle(indices)
        
        epoch_loss = 0.0
        
        for idx in indices:
            x_i = X[idx]
            y_i = y[idx]
            
            # Forward Pass: calculate prediction for ONE single sample
            y_pred_i = w * x_i + b
            
            # Compute squared error loss for this single sample
            loss_i = (y_pred_i - y_i) ** 2
            
            # Accumulate loss value for printing
            epoch_loss += loss_i.item()
            
            # Backward Pass: calculate gradients (dLoss/dw and dLoss/db) for this single sample
            loss_i.backward()
            
            # Update parameters and clear gradients
            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad
                w.grad.zero_()
                b.grad.zero_()
                
        average_epoch_loss = epoch_loss / num_samples
        print(f"Epoch {epoch:2d}/{epochs} | Avg Loss: {average_epoch_loss:.6f} | w: {w.item():.4f} | b: {b.item():.4f}")
        
    return w.item(), b.item()

if __name__ == "__main__":
    TRUE_W = 2.5
    TRUE_B = 1.0
    
    print(f"Generating synthetic data with true w = {TRUE_W} and true b = {TRUE_B}...")
    X, y = generate_synthetic_data(TRUE_W, TRUE_B, num_samples=100)
    
    w_gd, b_gd = train_batch_gradient_descent(X, y, epochs=100, lr=0.1)
    w_sgd, b_sgd = train_stochastic_gradient_descent(X, y, epochs=10, lr=0.01)
    
    print("\n=== Final Comparison ===")
    print(f"True Values   : w = {TRUE_W:.4f}, b = {TRUE_B:.4f}")
    print(f"GD (Scratch)  : w = {w_gd:.4f}, b = {b_gd:.4f}")
    print(f"SGD (Scratch) : w = {w_sgd:.4f}, b = {b_sgd:.4f}")
