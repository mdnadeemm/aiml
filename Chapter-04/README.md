# Chapter 4: Regression Techniques in Machine Learning

## Learning Objectives
By the end of this chapter, students will be able to:
- Understand fundamental concepts of regression analysis
- Implement linear and polynomial regression from scratch
- Apply regularization techniques to prevent overfitting
- Evaluate regression models using appropriate metrics
- Handle real-world regression challenges

## Table of Contents
1. [Introduction to Regression](#introduction)
2. [Linear Regression](#linear-regression)
3. [Polynomial Regression](#polynomial-regression)
4. [Regularization Techniques](#regularization)
5. [Model Evaluation](#evaluation)

## 1. Introduction to Regression {#introduction}

Regression analysis is a fundamental statistical and machine learning technique used to model relationships between variables and make predictions of continuous target values.

### What is Regression?

Regression seeks to establish mathematical relationships between:
- **Dependent Variable (Target)**: What we want to predict
- **Independent Variables (Features)**: What we use to make predictions

### Key Applications

**Finance**: Stock price prediction, risk assessment, portfolio optimization
**Healthcare**: Drug dosage modeling, treatment effectiveness analysis
**Marketing**: Sales forecasting, price optimization, customer value prediction
**Engineering**: Quality control, performance optimization, reliability analysis

### Types of Regression

**Simple Regression**: One feature predicts one target
**Multiple Regression**: Multiple features predict one target
**Polynomial Regression**: Non-linear relationships using polynomial terms

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import warnings
warnings.filterwarnings('ignore')

class RegressionDemo:
    """Comprehensive regression demonstration"""
    
    def __init__(self):
        # Generate sample data
        np.random.seed(42)
        self.generate_datasets()
    
    def generate_datasets(self):
        """Generate different types of sample datasets"""
        n = 100
        
        # Linear relationship
        self.X_linear = np.random.uniform(0, 10, n)
        self.y_linear = 2 * self.X_linear + 1 + np.random.normal(0, 1, n)
        
        # Non-linear relationship
        self.X_nonlinear = np.random.uniform(-3, 3, n)
        self.y_nonlinear = self.X_nonlinear**2 + np.random.normal(0, 0.5, n)
        
        # Multiple features
        self.X_multi = np.random.randn(n, 3)
        self.y_multi = (2*self.X_multi[:, 0] - 1.5*self.X_multi[:, 1] + 
                       0.8*self.X_multi[:, 2] + np.random.normal(0, 0.3, n))
    
    def demonstrate_regression_types(self):
        """Show different types of regression relationships"""
        print("=== REGRESSION RELATIONSHIP TYPES ===")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Linear relationship
        axes[0].scatter(self.X_linear, self.y_linear, alpha=0.6)
        axes[0].set_title('Linear Relationship')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('y')
        axes[0].grid(True, alpha=0.3)
        
        # Non-linear relationship
        axes[1].scatter(self.X_nonlinear, self.y_nonlinear, alpha=0.6)
        axes[1].set_title('Non-linear (Quadratic) Relationship')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('y')
        axes[1].grid(True, alpha=0.3)
        
        # Multiple features correlation
        correlation_matrix = np.corrcoef(np.column_stack([self.X_multi.T, self.y_multi]).T)
        im = axes[2].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[2].set_title('Multiple Features\nCorrelation Matrix')
        axes[2].set_xticks(range(4))
        axes[2].set_yticks(range(4))
        axes[2].set_xticklabels(['X1', 'X2', 'X3', 'y'])
        axes[2].set_yticklabels(['X1', 'X2', 'X3', 'y'])
        
        # Add correlation values to heatmap
        for i in range(4):
            for j in range(4):
                axes[2].text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                           ha='center', va='center', color='black')
        
        plt.colorbar(im, ax=axes[2])
        plt.tight_layout()
        plt.show()

demo = RegressionDemo()
demo.demonstrate_regression_types()
```

## 2. Linear Regression {#linear-regression}

Linear regression assumes a linear relationship between features and target: **y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε**

### Mathematical Foundation

**Objective**: Find coefficients (β) that minimize prediction errors
**Method**: Minimize Sum of Squared Errors (SSE)
**Solution**: Normal equation or gradient descent

### Implementation Approaches

**Normal Equation**: Direct mathematical solution β = (XᵀX)⁻¹Xᵀy
**Gradient Descent**: Iterative optimization for large datasets

```python
class LinearRegressionFromScratch:
    """Linear regression implementation from scratch"""
    
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.fitted = False
    
    def fit_normal_equation(self, X, y):
        """Fit using normal equation"""
        # Add intercept term
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # Normal equation: θ = (X^T X)^(-1) X^T y
        theta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        
        self.intercept = theta[0]
        self.coefficients = theta[1:]
        self.fitted = True
        
        print(f"Intercept: {self.intercept:.4f}")
        print(f"Coefficients: {self.coefficients}")
    
    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """Fit using gradient descent"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        m, n = X.shape
        theta = np.zeros(n + 1)
        X_with_intercept = np.column_stack([np.ones(m), X])
        
        cost_history = []
        
        for i in range(iterations):
            # Predictions
            predictions = X_with_intercept @ theta
            
            # Cost (MSE)
            cost = np.mean((predictions - y) ** 2)
            cost_history.append(cost)
            
            # Gradients
            gradients = (2/m) * X_with_intercept.T @ (predictions - y)
            
            # Update parameters
            theta -= learning_rate * gradients
            
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost:.6f}")
        
        self.intercept = theta[0]
        self.coefficients = theta[1:]
        self.fitted = True
        
        return cost_history
    
    def predict(self, X):
        """Make predictions"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return self.intercept + X @ self.coefficients
    
    def evaluate(self, X, y):
        """Calculate performance metrics"""
        predictions = self.predict(X)
        
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        
        return {'MSE': mse, 'RMSE': rmse, 'R²': r2}

# Demonstrate linear regression
print("=== LINEAR REGRESSION DEMONSTRATION ===")

# Simple linear regression
lr_simple = LinearRegressionFromScratch()
lr_simple.fit_normal_equation(demo.X_linear, demo.y_linear)

# Evaluate performance
metrics = lr_simple.evaluate(demo.X_linear, demo.y_linear)
print(f"\nPerformance Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Visualize results
plt.figure(figsize=(10, 6))
X_plot = np.linspace(demo.X_linear.min(), demo.X_linear.max(), 100)
y_plot = lr_simple.predict(X_plot)

plt.scatter(demo.X_linear, demo.y_linear, alpha=0.6, label='Data')
plt.plot(X_plot, y_plot, 'r-', linewidth=2, label='Linear Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Multiple linear regression with gradient descent
print("\n=== MULTIPLE LINEAR REGRESSION ===")
lr_multi = LinearRegressionFromScratch()
cost_history = lr_multi.fit_gradient_descent(demo.X_multi, demo.y_multi, learning_rate=0.1)

metrics_multi = lr_multi.evaluate(demo.X_multi, demo.y_multi)
print(f"\nMultiple Regression Performance:")
for metric, value in metrics_multi.items():
    print(f"{metric}: {value:.4f}")

# Plot cost function
plt.figure(figsize=(10, 6))
plt.plot(cost_history)
plt.title('Cost Function During Training')
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.grid(True, alpha=0.3)
plt.show()
```

## 3. Polynomial Regression {#polynomial-regression}

Polynomial regression extends linear regression to capture non-linear relationships by adding polynomial terms.

### Mathematical Foundation

Transform features: **x → [1, x, x², x³, ...]**
Then apply linear regression: **y = β₀ + β₁x + β₂x² + β₃x³ + ...**

### Advantages and Challenges

**Advantages**: Can model complex non-linear relationships
**Challenges**: Risk of overfitting with high-degree polynomials

```python
class PolynomialRegressionDemo:
    """Polynomial regression implementation and demonstration"""
    
    def __init__(self):
        self.models = {}
    
    def create_polynomial_features(self, X, degree):
        """Create polynomial features"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        if n_features != 1:
            raise ValueError("Polynomial features only supported for single feature")
        
        X_poly = np.ones((n_samples, degree + 1))
        for i in range(1, degree + 1):
            X_poly[:, i] = X[:, 0] ** i
        
        return X_poly
    
    def fit_polynomial(self, X, y, degree):
        """Fit polynomial regression of given degree"""
        X_poly = self.create_polynomial_features(X, degree)
        
        # Use normal equation
        theta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
        
        return theta
    
    def predict_polynomial(self, X, theta):
        """Make predictions with polynomial model"""
        degree = len(theta) - 1
        X_poly = self.create_polynomial_features(X, degree)
        return X_poly @ theta
    
    def compare_polynomial_degrees(self, X, y, max_degree=6):
        """Compare different polynomial degrees"""
        print("=== POLYNOMIAL REGRESSION COMPARISON ===")
        
        degrees = range(1, max_degree + 1)
        train_scores = []
        
        # Fit models for different degrees
        for degree in degrees:
            theta = self.fit_polynomial(X, y, degree)
            predictions = self.predict_polynomial(X, theta)
            r2 = r2_score(y, predictions)
            train_scores.append(r2)
            
            self.models[degree] = theta
            print(f"Degree {degree}: R² = {r2:.4f}")
        
        # Visualize results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        X_plot = np.linspace(X.min(), X.max(), 100)
        
        for i, degree in enumerate(degrees):
            if i < 6:  # Only plot first 6
                theta = self.models[degree]
                y_plot = self.predict_polynomial(X_plot, theta)
                
                axes[i].scatter(X, y, alpha=0.6, label='Data')
                axes[i].plot(X_plot, y_plot, 'r-', linewidth=2, 
                           label=f'Degree {degree}')
                axes[i].set_title(f'Polynomial Degree {degree}\nR² = {train_scores[i]:.3f}')
                axes[i].set_xlabel('X')
                axes[i].set_ylabel('y')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Plot R² vs degree
        plt.figure(figsize=(10, 6))
        plt.plot(degrees, train_scores, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Polynomial Degree')
        plt.ylabel('R² Score')
        plt.title('Model Performance vs Polynomial Degree')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return train_scores

# Demonstrate polynomial regression
poly_demo = PolynomialRegressionDemo()
poly_scores = poly_demo.compare_polynomial_degrees(demo.X_nonlinear, demo.y_nonlinear)

print("\nKey Observations:")
print("• Higher degrees can fit data better but may overfit")
print("• Optimal degree balances bias and variance")
print("• Very high degrees create unstable models")
```

## 4. Regularization Techniques {#regularization}

Regularization prevents overfitting by adding penalty terms to the cost function.

### Ridge Regression (L2 Regularization)

Adds L2 penalty: **Cost = MSE + α∑βᵢ²**
**Effect**: Shrinks coefficients toward zero, handles multicollinearity

### Lasso Regression (L1 Regularization)

Adds L1 penalty: **Cost = MSE + α∑|βᵢ|**
**Effect**: Can set coefficients to exactly zero, performs feature selection

### Elastic Net

Combines L1 and L2: **Cost = MSE + α₁∑|βᵢ| + α₂∑βᵢ²**
**Effect**: Balances Ridge and Lasso benefits

```python
class RegularizedRegression:
    """Demonstration of regularization techniques"""
    
    def __init__(self):
        self.models = {}
    
    def generate_high_dimensional_data(self, n_samples=100, n_features=50, n_informative=5):
        """Generate high-dimensional data for regularization demo"""
        np.random.seed(42)
        
        # Create feature matrix
        X = np.random.randn(n_samples, n_features)
        
        # Create target with only first few features being informative
        true_coeffs = np.zeros(n_features)
        true_coeffs[:n_informative] = np.random.randn(n_informative) * 2
        
        y = X @ true_coeffs + np.random.normal(0, 0.1, n_samples)
        
        return X, y, true_coeffs
    
    def compare_regularization_methods(self):
        """Compare different regularization approaches"""
        print("=== REGULARIZATION COMPARISON ===")
        
        # Generate high-dimensional data
        X, y, true_coeffs = self.generate_high_dimensional_data()
        
        # Split into train/test
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Different regularization strengths
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
        
        methods = {
            'Ridge': Ridge,
            'Lasso': Lasso
        }
        
        results = {}
        
        for method_name, Method in methods.items():
            results[method_name] = {'train_scores': [], 'test_scores': [], 'n_features': []}
            
            for alpha in alphas:
                model = Method(alpha=alpha)
                model.fit(X_train, y_train)
                
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                n_nonzero = np.sum(np.abs(model.coef_) > 1e-5)
                
                results[method_name]['train_scores'].append(train_score)
                results[method_name]['test_scores'].append(test_score)
                results[method_name]['n_features'].append(n_nonzero)
                
                print(f"{method_name} (α={alpha}): Train R²={train_score:.3f}, "
                      f"Test R²={test_score:.3f}, Features={n_nonzero}")
        
        # Visualize results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Training scores
        for method_name in methods.keys():
            axes[0].plot(alphas, results[method_name]['train_scores'], 'o-', 
                        label=method_name, linewidth=2)
        axes[0].set_xlabel('Regularization Strength (α)')
        axes[0].set_ylabel('Training R²')
        axes[0].set_title('Training Performance')
        axes[0].set_xscale('log')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Test scores
        for method_name in methods.keys():
            axes[1].plot(alphas, results[method_name]['test_scores'], 'o-', 
                        label=method_name, linewidth=2)
        axes[1].set_xlabel('Regularization Strength (α)')
        axes[1].set_ylabel('Test R²')
        axes[1].set_title('Test Performance')
        axes[1].set_xscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Number of features
        for method_name in methods.keys():
            axes[2].plot(alphas, results[method_name]['n_features'], 'o-', 
                        label=method_name, linewidth=2)
        axes[2].set_xlabel('Regularization Strength (α)')
        axes[2].set_ylabel('Number of Non-zero Features')
        axes[2].set_title('Feature Selection')
        axes[2].set_xscale('log')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    def demonstrate_coefficient_paths(self):
        """Show how coefficients change with regularization"""
        X, y, true_coeffs = self.generate_high_dimensional_data()
        
        alphas = np.logspace(-3, 2, 50)
        
        # Ridge path
        ridge_coefs = []
        lasso_coefs = []
        
        for alpha in alphas:
            # Ridge
            ridge_model = Ridge(alpha=alpha)
            ridge_model.fit(X, y)
            ridge_coefs.append(ridge_model.coef_)
            
            # Lasso
            lasso_model = Lasso(alpha=alpha)
            lasso_model.fit(X, y)
            lasso_coefs.append(lasso_model.coef_)
        
        ridge_coefs = np.array(ridge_coefs)
        lasso_coefs = np.array(lasso_coefs)
        
        # Plot coefficient paths
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Ridge paths (show first 10 features)
        for i in range(min(10, ridge_coefs.shape[1])):
            axes[0].plot(alphas, ridge_coefs[:, i], linewidth=2)
        axes[0].set_xlabel('Regularization Strength (α)')
        axes[0].set_ylabel('Coefficient Value')
        axes[0].set_title('Ridge Regression: Coefficient Paths')
        axes[0].set_xscale('log')
        axes[0].grid(True, alpha=0.3)
        
        # Lasso paths (show first 10 features)
        for i in range(min(10, lasso_coefs.shape[1])):
            axes[1].plot(alphas, lasso_coefs[:, i], linewidth=2)
        axes[1].set_xlabel('Regularization Strength (α)')
        axes[1].set_ylabel('Coefficient Value')
        axes[1].set_title('Lasso Regression: Coefficient Paths')
        axes[1].set_xscale('log')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Demonstrate regularization
reg_demo = RegularizedRegression()
reg_results = reg_demo.compare_regularization_methods()
reg_demo.demonstrate_coefficient_paths()

print("\nRegularization Key Points:")
print("• Ridge shrinks coefficients but keeps all features")
print("• Lasso can eliminate features by setting coefficients to zero")
print("• Higher α means stronger regularization")
print("• Optimal α balances bias and variance")
```

## Summary

This chapter covered essential regression techniques:

1. **Regression Fundamentals**: Understanding relationships and prediction
2. **Linear Regression**: Foundation method with normal equation and gradient descent
3. **Polynomial Regression**: Capturing non-linear relationships
4. **Regularization**: Preventing overfitting with Ridge and Lasso

### Key Takeaways:
- Regression predicts continuous values using feature relationships
- Linear regression is interpretable and computationally efficient
- Polynomial features can model non-linear relationships
- Regularization prevents overfitting in high-dimensional problems
- Model complexity should be balanced with generalization ability

### Best Practices:
- Check regression assumptions before applying models
- Use cross-validation to select optimal regularization parameters
- Analyze residuals to validate model adequacy
- Consider feature scaling for regularized methods
- Balance model complexity with interpretability needs

---

## Exercises

1. **Implementation**: Build polynomial regression with cross-validation
2. **Real Data**: Apply regression techniques to a real dataset
3. **Comparison**: Compare all methods on the same problem
4. **Feature Engineering**: Create interaction terms and test their impact
5. **Regularization Tuning**: Implement grid search for optimal parameters

---

*Regression techniques form the foundation for many machine learning applications. Master these methods to build effective predictive models.* 