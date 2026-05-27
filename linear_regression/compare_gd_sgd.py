import torch
import random
import matplotlib.pyplot as plt
import numpy as np

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
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

def generate_data(w_true, b_true, num_samples=100):
    X = torch.rand(num_samples) * 4 - 2
    noise = torch.randn(num_samples) * 0.4
    y = w_true * X + b_true + noise
    return X, y

def run_gd(X, y, epochs=100, lr=0.1):
    # torch.tensor(list, requires_grad=True): Converts a Python list into a PyTorch Tensor
    # and turns on gradient tracking.
    # We initialize parameters w and b far away from true values to visualize the trajectory.
    w = torch.tensor([-1.5], requires_grad=True)
    b = torch.tensor([-1.0], requires_grad=True)
    
    loss_history = []
    w_history = []
    b_history = []
    
    for epoch in range(epochs):
        # w.item(): Extracts the numerical float value out of the weight tensor.
        w_history.append(w.item())
        b_history.append(b.item())
        
        y_pred = w * X + b
        loss = torch.mean((y_pred - y) ** 2)
        loss_history.append(loss.item())
        
        # Calculate gradients (dLoss/dw and dLoss/db) via autograd.
        loss.backward()
        
        # Update parameters manually without tracking updates in autograd history.
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
            
    return w_history, b_history, loss_history

def run_sgd(X, y, epochs=100, lr=0.01):
    w = torch.tensor([-1.5], requires_grad=True)
    b = torch.tensor([-1.0], requires_grad=True)
    
    loss_history = []
    w_history = []
    b_history = []
    
    num_samples = len(X)
    
    for epoch in range(epochs):
        w_history.append(w.item())
        b_history.append(b.item())
        
        # torch.no_grad(): Temporarily disables gradient calculation.
        # We use this here to compute the loss over the whole dataset for logging purposes,
        # without adding this evaluation step to PyTorch's gradient calculation graph.
        with torch.no_grad():
            full_y_pred = w * X + b
            epoch_loss = torch.mean((full_y_pred - y) ** 2).item()
            loss_history.append(epoch_loss)
            
        indices = list(range(num_samples))
        random.shuffle(indices)
        
        for idx in indices:
            x_i = X[idx]
            y_i = y[idx]
            
            y_pred_i = w * x_i + b
            loss_i = (y_pred_i - y_i) ** 2
            
            # Compute gradient for this single sample
            loss_i.backward()
            
            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad
                w.grad.zero_()
                b.grad.zero_()
                
    return w_history, b_history, loss_history

if __name__ == "__main__":
    TRUE_W = 2.5
    TRUE_B = 1.0
    # Training length
    EPOCHS = 60
    
    print("Generating data...")
    X, y = generate_data(TRUE_W, TRUE_B, num_samples=100)
    
    print(f"Running Batch Gradient Descent (GD) for {EPOCHS} epochs...")
    w_gd, b_gd, loss_gd = run_gd(X, y, epochs=EPOCHS, lr=0.1)
    
    print(f"Running Stochastic Descent (SGD) for {EPOCHS} epochs...")
    w_sgd, b_sgd, loss_sgd = run_sgd(X, y, epochs=EPOCHS, lr=0.01)
    
    print("Generating comparison visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Gradient Descent (GD) vs. Stochastic Gradient Descent (SGD) in PyTorch", 
                 fontsize=18, fontweight='bold', color='#f8fafc', y=1.02)
    
    # --- SUBPLOT 1: Loss curves ---
    ax_loss = axes[0]
    ax_loss.plot(loss_gd, label='Batch GD (lr=0.1)', color='#6366f1', linewidth=2.5)
    ax_loss.plot(loss_sgd, label='Stochastic SGD (lr=0.01)', color='#10b981', linewidth=2.5, linestyle='--')
    ax_loss.set_title("Loss Convergence (MSE)", fontsize=14, fontweight='semibold', pad=12, color='#f1f5f9')
    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Loss", fontsize=12)
    ax_loss.grid(True)
    ax_loss.legend(facecolor='#1e1e24', edgecolor='#334155', fontsize=11)
    
    # --- SUBPLOT 2: Regression Lines Fit ---
    ax_fit = axes[1]
    # X.numpy() and y.numpy(): Converts PyTorch Tensors back into Standard NumPy Arrays.
    # Matplotlib does not always support plotting PyTorch Tensors directly, 
    # so converting them to NumPy is a necessary step for clean plotting.
    x_np = X.numpy()
    y_np = y.numpy()
    
    ax_fit.scatter(x_np, y_np, color='#64748b', alpha=0.6, label='Noisy Data Points', edgecolors='none', s=35)
    
    # Line points
    line_x = np.linspace(-2, 2, 100)
    true_line = TRUE_W * line_x + TRUE_B
    gd_line = w_gd[-1] * line_x + b_gd[-1]
    sgd_line = w_sgd[-1] * line_x + b_sgd[-1]
    
    ax_fit.plot(line_x, true_line, label='True Function', color='#ef4444', linewidth=2.5)
    ax_fit.plot(line_x, gd_line, label='Batch GD Fit', color='#6366f1', linewidth=2, linestyle=':')
    ax_fit.plot(line_x, sgd_line, label='Stochastic SGD Fit', color='#10b981', linewidth=2, linestyle='-.')
    
    ax_fit.set_title("Fitted Regression Lines", fontsize=14, fontweight='semibold', pad=12, color='#f1f5f9')
    ax_fit.set_xlabel("Input (X)", fontsize=12)
    ax_fit.set_ylabel("Output (y)", fontsize=12)
    ax_fit.grid(True)
    ax_fit.legend(facecolor='#1e1e24', edgecolor='#334155', fontsize=11)
    
    # --- SUBPLOT 3: Parameter Trajectory Space ---
    ax_traj = axes[2]
    # Plot GD path
    ax_traj.plot(w_gd, b_gd, color='#6366f1', marker='o', markersize=4, alpha=0.8, linewidth=1.8, label='Batch GD path')
    # Plot SGD path
    ax_traj.plot(w_sgd, b_sgd, color='#10b981', marker='x', markersize=5, alpha=0.8, linewidth=1.5, label='Stochastic SGD path')
    
    # Highlight start, end, and target
    ax_traj.scatter(w_gd[0], b_gd[0], color='#ef4444', s=100, zorder=5, label='Start (-1.5, -1.0)', marker='s')
    ax_traj.scatter(TRUE_W, TRUE_B, color='#eab308', s=150, zorder=5, label='Target (2.5, 1.0)', marker='*')
    
    # Add arrows to indicate direction of updates for GD (every 10 steps)
    for i in range(0, len(w_gd)-10, 10):
        ax_traj.annotate('', xy=(w_gd[i+5], b_gd[i+5]), xytext=(w_gd[i], b_gd[i]),
                         arrowprops=dict(arrowstyle="->", color='#818cf8', lw=1.5))
                         
    ax_traj.set_title("Parameter Trajectory (w vs b)", fontsize=14, fontweight='semibold', pad=12, color='#f1f5f9')
    ax_traj.set_xlabel("Weight (w)", fontsize=12)
    ax_traj.set_ylabel("Bias (b)", fontsize=12)
    ax_traj.grid(True)
    ax_traj.legend(facecolor='#1e1e24', edgecolor='#334155', fontsize=10, loc='lower right')
    
    # Save plot
    plt.tight_layout()
    plot_path = "gd_vs_sgd_comparison.png"
    plt.savefig(plot_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved successfully to: {plot_path}")
    print(f"GD final parameters: w = {w_gd[-1]:.4f}, b = {b_gd[-1]:.4f}")
    print(f"SGD final parameters: w = {w_sgd[-1]:.4f}, b = {b_sgd[-1]:.4f}")
