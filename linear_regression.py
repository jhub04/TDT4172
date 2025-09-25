import numpy as np

class LinearRegression():
    
    def __init__(self, learning_rate=0.001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs 
        self.weights, self.bias = None, None
        

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
            y_pred = (X @ self.weights) + self.bias

            # Bruker Mean Squared Error (MSE) som tapsfunksjon
            grad_w = (2/n_samples) * np.dot(X.T, (y_pred - y))         
            grad_b = (2/n_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.learning_rate * grad_w
            self.bias = self.bias - self.learning_rate * grad_b
            
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
        
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred









