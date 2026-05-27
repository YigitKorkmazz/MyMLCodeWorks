# Practicing Linear Regression, GD, and SGD in PyTorch

Welcome! This workspace is designed to help you practice and understand Linear Regression, Batch Gradient Descent (GD), and Stochastic Gradient Descent (SGD) in PyTorch. 

We implement linear regression in two ways:
1. **From Scratch**: Using raw PyTorch tensors and manual weight updates, mirroring the math exactly.
2. **The PyTorch Way**: Using PyTorch's native high-level abstractions (`torch.nn` and `torch.optim`).

---

## 1. Mathematical Foundation

### Linear Regression Model
A linear regression model assumes a linear relationship between input features $x$ and output target $y$:
$$\hat{y} = w \cdot x + b$$
where:
* $\hat{y}$ is the predicted value.
* $w$ is the weight (slope).
* $b$ is the bias (y-intercept).

### Mean Squared Error (MSE) Loss
To measure how well our model fits the data, we use the Mean Squared Error (MSE) cost function:
$$L(w, b) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 = \frac{1}{N} \sum_{i=1}^{N} (y_i - (w \cdot x_i + b))^2$$
where $N$ is the number of samples in our dataset.

---

## 2. Gradient Descent (GD) vs. Stochastic Gradient Descent (SGD)

To find the optimal parameters $w$ and $b$ that minimize the loss $L$, we use optimization algorithms.

### A. Batch Gradient Descent (GD)
In Batch Gradient Descent, we calculate the gradients using the **entire dataset** at each step:
$$\frac{\partial L}{\partial w} = -\frac{2}{N} \sum_{i=1}^{N} x_i (y_i - \hat{y}_i)$$
$$\frac{\partial L}{\partial b} = -\frac{2}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)$$

The parameter update rule (where $\eta$ is the learning rate) is:
$$w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}$$
$$b \leftarrow b - \eta \cdot \frac{\partial L}{\partial b}$$

* **Pros**: Stable convergence, smooth path to the minimum.
* **Cons**: Extremely slow and memory-intensive for large datasets because we process all samples before taking a single step.

### B. Stochastic Gradient Descent (SGD)
In Stochastic Gradient Descent, we compute the gradient and update parameters using **only one random sample** $(x_i, y_i)$ at a time:
$$\frac{\partial L}{\partial w} = -2 \cdot x_i (y_i - \hat{y}_i)$$
$$\frac{\partial L}{\partial b} = -2 \cdot (y_i - \hat{y}_i)$$

The parameter update rule is applied per sample:
$$w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}$$
$$b \leftarrow b - \eta \cdot \frac{\partial L}{\partial b}$$

* **Pros**: Much faster updates, uses very little memory, can escape local minima due to its noisy/fluctuating nature.
* **Cons**: The path to the minimum is noisy and zig-zags; it may bounce around the minimum instead of converging precisely.

---

## 3. PyTorch Autograd Concepts

When using PyTorch, you don't need to manually compute derivatives ($\frac{\partial L}{\partial w}$, $\frac{\partial L}{\partial b}$). PyTorch's **Autograd** engine handles this via a computational graph.

### Key Concepts to Learn:

1. **`requires_grad=True`**:
   Tells PyTorch to track all mathematical operations on this tensor. PyTorch will build a computational graph dynamically during the forward pass.
   ```python
   w = torch.randn(1, requires_grad=True)
   ```

2. **`.backward()` (The Backpropagation Trigger)**:
   When you call `loss.backward()`, PyTorch traverses the computational graph backward from the `loss` tensor to calculate the gradient of `loss` with respect to all leaf tensors that have `requires_grad=True`. The computed gradients are stored in the `.grad` attribute of those tensors (e.g., `w.grad` and `b.grad`).

3. **`with torch.no_grad():` (Disabling Tracking)**:
   When updating parameters manually ($w = w - \eta \cdot w.grad$), we wrap the updates in a `with torch.no_grad():` block. If we don't, PyTorch will track the update operation itself, adding it to the computational graph and leading to runtime errors or incorrect gradient accumulations.
   ```python
   with torch.no_grad():
       w -= lr * w.grad
   ```

4. **Gradient Accumulation & Zeroing (`.grad.zero_()`)**:
   By default, PyTorch **accumulates** gradients (adds them together) on subsequent calls to `.backward()`. It does not overwrite them. Therefore, you **MUST** zero the gradients after updating the parameters, before the next training step:
   ```python
   w.grad.zero_()
   ```

---

## 4. File Structure

* **[linear_regression_scratch.py](file:///Users/yusufyigitkorkmaz/Desktop/MyMLCodeWorks/linear_regression_scratch.py)**: Code that generates synthetic data and trains a linear model from scratch (manual loop, manual gradient updates) using both GD and SGD.
* **[linear_regression_pytorch.py](file:///Users/yusufyigitkorkmaz/Desktop/MyMLCodeWorks/linear_regression_pytorch.py)**: Code that trains the same model using `torch.nn.Module`, `torch.nn.MSELoss`, and `torch.optim.SGD`.
* **[compare_gd_sgd.py](file:///Users/yusufyigitkorkmaz/Desktop/MyMLCodeWorks/compare_gd_sgd.py)**: Runs training comparisons and generates beautiful, insightful plots comparing convergence speed, parameter path trajectories, and the regression fit line.

---

## 5. Getting Started & Running Code

First, ensure your virtual environment is active:
```bash
source .venv/bin/activate
```

### Run GD and SGD implemented from scratch:
```bash
python3 linear_regression_scratch.py
```

### Run using standard PyTorch modules:
```bash
python3 linear_regression_pytorch.py
```

### Run the comparison script to generate plots:
```bash
python3 compare_gd_sgd.py
```
This will generate `gd_vs_sgd_comparison.png` in your workspace, displaying the training dynamics.
