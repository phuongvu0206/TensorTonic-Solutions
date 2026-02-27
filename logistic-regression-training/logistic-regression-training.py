import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def _bce_loss(y, prob_y_pred):
    eps = 1e-15
    prob_y_pred = np.clip(prob_y_pred, eps, 1 - eps)
    return -np.mean(y*np.log(prob_y_pred) + (1-y)*np.log(1-prob_y_pred))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n_samples, n_features = X.shape
    
    w = np.zeros(n_features)
    b = 0
    losses = []
    
    for i in range(steps):
        linear_model = X @ w + b       
        probs = _sigmoid(linear_model)
        loss = _bce_loss(y, probs)
        losses.append(loss)

        dw = 1/n_samples * (X.T @ (probs - y))
        db = 1/n_samples * np.sum(probs - y)
        
        w = w - lr*dw
        b = b - lr*db

    return (w, b)
        
        





