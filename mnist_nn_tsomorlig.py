import numpy as np
from sklearn.datasets import fetch_openml

def sigmoid(z):
    """Standard sigmoid activation function with overflow protection."""
    z = np.clip(z, -500, 500) 
    return 1 / (1 + np.exp(-z))

def forward(X, W1, b1, W2, b2):
    """Performs the forward pass through the 2-layer network."""
    # Step 1: Hidden Layer (Linear + Tanh)
    Z1 = np.dot(X, W1) + b1 
    A1 = np.tanh(Z1)        
    
    # Step 2: Output Layer (Linear + Sigmoid)
    Z2 = np.dot(A1, W2) + b2 
    y_hat = sigmoid(Z2)      
    
    cache = (Z1, A1, Z2) 
    return y_hat, cache

def compute_loss(y_true, y_hat):
    """Calculates Binary Cross-Entropy loss."""
    eps = 1e-15 
    y_hat = np.clip(y_hat, eps, 1 - eps) 
    return -np.mean(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))

def backward(X, y, y_hat, cache, W2):
    """Performs backpropagation to calculate gradients."""
    n = X.shape[0]
    Z1, A1, Z2 = cache 
    
    # 1. Output Layer Error 
    delta2 = y_hat - y 
    
    # 2. Output Layer Gradients 
    dW2 = (1/n) * np.dot(A1.T, delta2)
    db2 = (1/n) * np.sum(delta2, axis=0)
    
    # 3. Hidden Layer Error 
    delta1 = np.dot(delta2, W2.T) * (1 - np.power(A1, 2))
    
    # 4. Hidden Layer Gradients 
    dW1 = (1/n) * np.dot(X.T, delta1)
    db1 = (1/n) * np.sum(delta1, axis=0)
    
    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads

def train(X, y, hidden_size=64, learning_rate=0.1, iterations=1000):
    """Initializes weights and runs the training loop."""
    W1 = np.random.randn(784, hidden_size).astype(np.float32) * 0.01
    b1 = np.zeros(hidden_size, dtype=np.float32)
    W2 = np.random.randn(hidden_size, 1).astype(np.float32) * 0.01
    b2 = np.zeros(1, dtype=np.float32)
    
    losses = []
    
    for i in range(iterations):
        y_hat, cache = forward(X, W1, b1, W2, b2)
        curr_loss = compute_loss(y, y_hat)
        losses.append(curr_loss)
        
        grads = backward(X, y, y_hat, cache, W2)
        
        # Update parameters 
        W1 -= learning_rate * grads['dW1']
        b1 -= learning_rate * grads['db1']
        W2 -= learning_rate * grads['dW2']
        b2 -= learning_rate * grads['db2']
        
        if i % 100 == 0:
            print(f"Iteration {i}: Loss {curr_loss:.4f}")
            
    return W1, b1, W2, b2, losses

if __name__ == "__main__":
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X_raw, y_raw = mnist.data.astype(np.float32), mnist.target.astype(int)

    # Preprocess: Normalize pixels to [0, 1] 
    X_raw /= 255.0

    # Binary Labels: 1 if even, 0 if odd 
    y_binary = (y_raw % 2 == 0).astype(np.float32).reshape(-1, 1)

    # Train/Test Split
    X_train, X_test = X_raw[:60000], X_raw[60000:]
    y_train, y_test = y_binary[:60000], y_binary[60000:]

    print("Starting training...")
    W1_f, b1_f, W2_f, b2_f, train_losses = train(
        X_train, y_train, hidden_size=64, learning_rate=0.5, iterations=1000
    )

    # Evaluation
    y_test_hat, _ = forward(X_test, W1_f, b1_f, W2_f, b2_f)
    predictions = (y_test_hat > 0.5).astype(np.float32)
    accuracy = np.mean(predictions == y_test)
    print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")