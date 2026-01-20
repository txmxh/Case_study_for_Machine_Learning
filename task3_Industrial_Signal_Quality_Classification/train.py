import pandas as pd
import numpy as np
from model import Model
import time

def train_model():
    print("Loading datasets...")
    try:
        train_df = pd.read_csv('datasets/train_df.csv')
        test_df = pd.read_csv('datasets/test_df.csv')
        submission = pd.read_csv('datasets/submission.csv')
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # --- 1. Data Prep ---
    feature_cols = [c for c in test_df.columns if c != 'index']
    target_col = list(set(train_df.columns) - set(feature_cols) - {'index'})[0]
    
    X = train_df[feature_cols].values
    y = train_df[target_col].values
    
    # --- FIX START: Handle Types and NaNs ---
    # 1. Check for NaNs in the target and drop those rows
    if np.isnan(y).any():
        print("Warning: NaNs found in target variable. Dropping those rows...")
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
    
    # 2. Convert to Integer (Critical for one-hot encoding)
    y = y.astype(int)
    # --- FIX END ---
    
    # Normalize
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    X_scaled = (X - mean) / std
    
    # One-hot encoding
    num_classes = 4
    y_enc = np.eye(num_classes)[y]

    # --- 2. Class Weighting ---
    class_counts = np.bincount(y)
    total_samples = len(y)
    # Add a small epsilon to avoid division by zero if a class is missing
    safe_counts = np.where(class_counts == 0, 1, class_counts) 
    class_weights = total_samples / (num_classes * safe_counts)
    
    print(f"Class Weights computed: {class_weights}")
    weights_vector = np.array([class_weights[label] for label in y])

    # --- 3. Hyperparameters ---
    input_dim = len(feature_cols)
    hidden1 = 256  
    hidden2 = 128 
    output_dim = 4
    
    lr = 0.001
    epochs = 40
    batch_size = 64
    reg = 0.0005
    dropout_p = 0.2

    # --- 4. Initialization (He Init) ---
    np.random.seed(42)
    params = {
        'W1': np.random.randn(input_dim, hidden1) * np.sqrt(2./input_dim),
        'b1': np.zeros(hidden1),
        'W2': np.random.randn(hidden1, hidden2) * np.sqrt(2./hidden1),
        'b2': np.zeros(hidden2),
        'W3': np.random.randn(hidden2, output_dim) * np.sqrt(2./hidden2),
        'b3': np.zeros(output_dim)
    }
    
    # Adam Params
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    print(f"Training Pro Model (Adam + Class Weights + Dropout)...")
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(X))
        X_sh = X_scaled[indices]
        y_sh = y_enc[indices]
        w_sh = weights_vector[indices]
        
        epoch_loss = 0
        
        for i in range(0, len(X), batch_size):
            X_batch = X_sh[i:i+batch_size]
            y_batch = y_sh[i:i+batch_size]
            w_batch = w_sh[i:i+batch_size]
            
            # Forward
            z1 = np.dot(X_batch, params['W1']) + params['b1']
            a1 = np.maximum(0, z1)
            mask1 = (np.random.rand(*a1.shape) > dropout_p) / (1 - dropout_p)
            a1 *= mask1
            
            z2 = np.dot(a1, params['W2']) + params['b2']
            a2 = np.maximum(0, z2)
            mask2 = (np.random.rand(*a2.shape) > dropout_p) / (1 - dropout_p)
            a2 *= mask2
            
            z3 = np.dot(a2, params['W3']) + params['b3']
            
            exp_z = np.exp(z3 - np.max(z3, axis=1, keepdims=True))
            probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            
            # Loss
            correct_logprobs = -np.log(probs[range(len(y_batch)), np.argmax(y_batch, axis=1)] + 1e-9)
            loss = np.mean(correct_logprobs * w_batch)
            epoch_loss += loss

            # Backprop
            delta3 = probs - y_batch
            delta3 *= w_batch[:, None] 
            delta3 /= len(X_batch)
            
            grads = {}
            grads['W3'] = np.dot(a2.T, delta3) + reg * params['W3']
            grads['b3'] = np.sum(delta3, axis=0)
            
            delta2 = np.dot(delta3, params['W3'].T) * (z2 > 0)
            delta2 *= mask2
            grads['W2'] = np.dot(a1.T, delta2) + reg * params['W2']
            grads['b2'] = np.sum(delta2, axis=0)
            
            delta1 = np.dot(delta2, params['W2'].T) * (z1 > 0)
            delta1 *= mask1
            grads['W1'] = np.dot(X_batch.T, delta1) + reg * params['W1']
            grads['b1'] = np.sum(delta1, axis=0)
            
            # Adam Update
            t = epoch * (len(X) // batch_size) + (i // batch_size) + 1
            for k in params:
                m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                v[k] = beta2 * v[k] + (1 - beta2) * grads[k]**2
                m_hat = m[k] / (1 - beta1**t)
                v_hat = v[k] / (1 - beta2**t)
                params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    # Save
    np.savez('model_weights.npz', **params, mean=mean, std=std)
    
    # Predict
    print("Generating submission...")
    model = Model()
    preds = model.predict(test_df)
    submission['Predicted'] = preds
    submission.to_csv('submission.csv', index=False)
    print("Done. Submission ready.")

if __name__ == "__main__":
    train_model()