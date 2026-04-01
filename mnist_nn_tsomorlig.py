import numpy as np
from sklearn.datasets import fetch_openml

# 1. Load MNIST dataset 
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_raw, y_raw = mnist.data.astype(np.float32), mnist.target.astype(int)

# 2. Preprocess: Normalize pixels to [0, 1] 
X_raw /= 255.0

# 3. Binary Labels: 1 if even (0,2,4,6,8), 0 if odd (1,3,5,7,9) 
y_binary = (y_raw % 2 == 0).astype(np.float32).reshape(-1, 1)

# 4. Train/Test Split: First 60k for training, last 10k for testing
X_train, X_test = X_raw[:60000], X_raw[60000:]
y_train, y_test = y_binary[:60000], y_binary[60000:]

def sigmoid(z):
    z = np.clip(z, -500, 500) # Prevent overflow 
    return 1 / (1 + np.exp(-z)) # Standard sigmoid formula

def forward(X, W1, b1, W2, b2):
    # Step 1: Hidden Layer (Linear + Tanh)
    Z1 = np.dot(X, W1) + b1 # (n, 784) * (784, 64) + (64,) -> (n, 64) 
    A1 = np.tanh(Z1)        # Apply non-linear activation 
    
    # Step 2: Output Layer (Linear + Sigmoid)
    Z2 = np.dot(A1, W2) + b2 # (n, 64) * (64, 1) + (1,) -> (n, 1)
    y_hat = sigmoid(Z2)      # Output probability in [0, 1] 
    
    cache = (Z1, A1, Z2) # Store for backpropagation 
    return y_hat, cache

def compute_loss(y_true, y_hat):
    eps = 1e-15 # Constant to avoid log(0)
    y_hat = np.clip(y_hat, eps, 1 - eps) # Clip values 
    # Calculate average loss across the batch 
    return -np.mean(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))

def backward(X, y, y_hat, cache, W2):
    n = X.shape[0] # Number of samples
    Z1, A1, Z2 = cache # Retrieve intermediate values
    
    # 1. Output Layer Error 
    delta2 = y_hat - y # Difference between prediction and truth (n, 1)
    
    # 2. Output Layer Gradients 
    dW2 = (1/n) * np.dot(A1.T, delta2) # Gradient for W2 (64, 1)
    db2 = (1/n) * np.sum(delta2, axis=0) # Gradient for b2 (1,)
    
    # 3. Hidden Layer Error 
    # Backpropagate error through W2 and tanh derivative (1 - A1^2) 
    delta1 = np.dot(delta2, W2.T) * (1 - np.power(A1, 2)) # (n, 64)
    
    # 4. Hidden Layer Gradients 
    dW1 = (1/n) * np.dot(X.T, delta1) # Gradient for W1 (784, 64)
    db1 = (1/n) * np.sum(delta1, axis=0) # Gradient for b1 (64,)
    
    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads

def train(X, y, hidden_size=64, learning_rate=0.1, iterations=1000):
    # Initialize parameters with small random values
    W1 = np.random.randn(784, hidden_size).astype(np.float32) * 0.01
    b1 = np.zeros(hidden_size, dtype=np.float32)
    W2 = np.random.randn(hidden_size, 1).astype(np.float32) * 0.01
    b2 = np.zeros(1, dtype=np.float32)
    
    losses = []
    
    for i in range(iterations):
        # Forward pass
        y_hat, cache = forward(X, W1, b1, W2, b2)
        
        # Compute and record loss
        curr_loss = compute_loss(y, y_hat)
        losses.append(curr_loss)
        
        # Backward pass
        grads = backward(X, y, y_hat, cache, W2)
        
        # Update parameters 
        W1 -= learning_rate * grads['dW1']
        b1 -= learning_rate * grads['db1']
        W2 -= learning_rate * grads['dW2']
        b2 -= learning_rate * grads['db2']
        
        # Optional: Print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Loss {curr_loss:.4f}")
            
    return W1, b1, W2, b2, losses

# # Run training with standard hyperparameters 
# W1_final, b1_final, W2_final, b2_final, train_losses = train(
#     X_train, y_train, hidden_size=64, learning_rate=0.5, iterations=1000
# )
# # Run training with standard hyperparameters 
# W1_final, b1_final, W2_final, b2_final, train_losses = train(
#     X_train, y_train, hidden_size=64, learning_rate=0.5, iterations=1000
# )

# # Test the model
# y_test_hat, _ = forward(X_test, W1_final, b1_final, W2_final, b2_final)
# predictions = (y_test_hat > 0.5).astype(np.float32) # Convert to 0/1 
# accuracy = np.mean(predictions == y_test)
# print(f"Final Test Accuracy: {accuracy * 100:.2f}%") # Expected 90-95% 
# # Test the model
# y_test_hat, _ = forward(X_test, W1_final, b1_final, W2_final, b2_final)
# predictions = (y_test_hat > 0.5).astype(np.float32) # Convert to 0/1 
# accuracy = np.mean(predictions == y_test)
# print(f"Final Test Accuracy: {accuracy * 100:.2f}%") # Expected 90-95% 

import numpy as np
import tracemalloc

def run_experiment(X_train, y_train, X_test, y_test, hidden_size, learning_rate, iterations=1000):
    # ---- Helpers ----
    def compute_accuracy(y_hat, y_true):
        preds = (y_hat > 0.5).astype(np.float32)
        return np.mean(preds == y_true)

    def compute_loss(y_hat, y_true):
        eps = 1e-8
        return -np.mean(y_true * np.log(y_hat + eps) + (1 - y_true) * np.log(1 - y_hat + eps))

    def gradient_check_single(X, y, W1, b1, W2, b2, eps=1e-5):
        i, j = 0, 0

        original = W1[i, j]

        W1[i, j] = original + eps
        y_hat_plus, _ = forward(X, W1, b1, W2, b2)
        loss_plus = compute_loss(y_hat_plus, y)

        W1[i, j] = original - eps
        y_hat_minus, _ = forward(X, W1, b1, W2, b2)
        loss_minus = compute_loss(y_hat_minus, y)

        W1[i, j] = original

        numerical_grad = (loss_plus - loss_minus) / (2 * eps)
        return abs(numerical_grad)

    # ---- Memory tracking ----
    tracemalloc.start()

    # ---- Train ----
    W1, b1, W2, b2, train_losses = train(
        X_train, y_train,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        iterations=iterations
    )

    # ---- Train metrics ----
    y_train_hat, _ = forward(X_train, W1, b1, W2, b2)
    train_loss = compute_loss(y_train_hat, y_train)
    train_acc = compute_accuracy(y_train_hat, y_train)

    # ---- Test metrics ----
    y_test_hat, _ = forward(X_test, W1, b1, W2, b2)
    test_acc = compute_accuracy(y_test_hat, y_test)

    # ---- Memory ----
    current, peak = tracemalloc.get_traced_memory()
    peak_memory_mb = peak / (1024 * 1024)
    tracemalloc.stop()

    # ---- Gradient check ----
    gradient_check = {
        "W1_100_30": gradient_check_single(X_train[:100], y_train[:100], W1, b1, W2, b2),
        "W1_500_50": gradient_check_single(X_train[:500], y_train[:500], W1, b1, W2, b2),
        "W2_30_0": gradient_check_single(X_train[:30], y_train[:30], W1, b1, W2, b2),
        "b1_10": gradient_check_single(X_train[:10], y_train[:10], W1, b1, W2, b2)
    }

    return {
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        "peak_memory_mb": float(peak_memory_mb)
    }, {k: float(v) for k, v in gradient_check.items()}

learning_rates_tested = [0.1, 1.0]
hidden_sizes_tested = [64, 128, 256]

for hidden_size in hidden_sizes_tested:  
    for lr in learning_rates_tested: 
        print("hidden_size: ", hidden_size, ", learning_rate: ", lr)
        metrics, grad_check = run_experiment(
            X_train, y_train, X_test, y_test,
            hidden_size=hidden_size,
            learning_rate=lr
        )
        print(metrics)
        print("gradient_check:", grad_check)