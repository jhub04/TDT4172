import numpy as np

class LogisticRegression():

    def __init__(self, learning_rate=0.1, epochs=1000, threshold=0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []
        self.threshold = threshold

    def sigmoid_function(self, z):
        """
        Maps real numbers to the domain [0, 1]
        """
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, y, y_pred):
        """
        Calculates loss for the whole dataset based on binary cross entropy loss
        """
        loss = -(y*np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return np.mean(loss)

    def compute_gradients(self, X, y, y_pred):
        """
        Computes the gradients for weights and bias based on the loss function binary cross entropy (log loss)
        Returns gradient_weights, gradient_bias
        """
        r = (y_pred - y)
        grad_w = (X.T @ r) / X.shape[0]
        grad_b = -np.mean(y - y_pred)
        return grad_w, grad_b

    def update_parameters(self, grad_w, grad_b):
        self.weights = self.weights - self.learning_rate * grad_w
        self.bias = self.bias - self.learning_rate * grad_b


    def accuracy(self, true_values, predictions):
        return np.mean(true_values == predictions)
    
    def fit(self, X, y):
        """ 
        Estimates parameters for the classifier 
        
        Args: 
        X (array<m,n>): a matrix of floats with 
            m rows (#samples) and n columns (#features) 
        
        y (array<m>): a vector of floats 
        
        """

        X = np.array(X)
        y = np.array(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)  

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            lin_model = (X @ self.weights) + self.bias
            y_pred = self.sigmoid_function(lin_model)

            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            
            self.update_parameters(grad_w, grad_b)

            loss = self._compute_loss(y, y_pred)
            pred_to_class = [1 if _y > self.threshold else 0 for _y in y_pred]
            self.train_accuracies.append(self.accuracy(y, pred_to_class))
            self.losses.append(loss)



    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        lin_model = (X @ self.weights) + self.bias
        y_pred = self.sigmoid_function(lin_model)
        return [1 if _y > self.threshold else 0 for _y in y_pred]
    
    def predict_proba(self, X):
        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        lin_model = (X @ self.weights) + self.bias
        y_pred_prob = self.sigmoid_function(lin_model)
        return y_pred_prob







