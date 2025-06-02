# Chapter 3: Machine Learning Algorithms Overview

## Learning Objectives
By the end of this chapter, students will be able to:
- Understand the major families of machine learning algorithms
- Compare and contrast different algorithmic approaches
- Choose appropriate algorithms for specific problem types
- Implement basic versions of key algorithms from scratch
- Evaluate algorithm performance and characteristics

## Table of Contents
1. [Algorithm Taxonomy](#algorithm-taxonomy)
2. [Linear Algorithms](#linear-algorithms)
3. [Tree-Based Algorithms](#tree-based-algorithms)
4. [Instance-Based Algorithms](#instance-based-algorithms)
5. [Ensemble Methods](#ensemble-methods)
6. [Probabilistic Algorithms](#probabilistic-algorithms)
7. [Neural Network Algorithms](#neural-network-algorithms)
8. [Clustering Algorithms](#clustering-algorithms)
9. [Algorithm Selection Guide](#algorithm-selection)
10. [Performance Comparison](#performance-comparison)

## 1. Algorithm Taxonomy {#algorithm-taxonomy}

Machine learning algorithms can be categorized in multiple ways. Understanding these taxonomies helps in selecting the right algorithm for your problem.

### By Learning Type

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

class AlgorithmTaxonomy:
    """Visualize and understand ML algorithm taxonomy"""
    
    def __init__(self):
        self.taxonomy = {
            'Supervised Learning': {
                'Classification': [
                    'Logistic Regression', 'Decision Trees', 'Random Forest',
                    'SVM', 'K-Nearest Neighbors', 'Naive Bayes', 'Neural Networks'
                ],
                'Regression': [
                    'Linear Regression', 'Polynomial Regression', 'Ridge/Lasso',
                    'Decision Tree Regression', 'Random Forest Regression', 'SVR'
                ]
            },
            'Unsupervised Learning': {
                'Clustering': [
                    'K-Means', 'Hierarchical Clustering', 'DBSCAN',
                    'Gaussian Mixture Models', 'Mean Shift'
                ],
                'Dimensionality Reduction': [
                    'PCA', 'LDA', 't-SNE', 'UMAP', 'ICA'
                ],
                'Association Rules': [
                    'Apriori', 'FP-Growth', 'Eclat'
                ]
            },
            'Reinforcement Learning': {
                'Model-Free': [
                    'Q-Learning', 'SARSA', 'Policy Gradient', 'Actor-Critic'
                ],
                'Model-Based': [
                    'Value Iteration', 'Policy Iteration', 'Monte Carlo Tree Search'
                ]
            }
        }
    
    def visualize_taxonomy(self):
        """Create a hierarchical visualization of ML algorithms"""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Define colors for each learning type
        colors = {'Supervised Learning': 'lightblue', 
                 'Unsupervised Learning': 'lightgreen',
                 'Reinforcement Learning': 'lightcoral'}
        
        y_pos = 0.9
        for learning_type, categories in self.taxonomy.items():
            # Draw main category
            rect = patches.Rectangle((0.1, y_pos-0.05), 0.8, 0.08, 
                                   linewidth=2, edgecolor='black', 
                                   facecolor=colors[learning_type], alpha=0.7)
            ax.add_patch(rect)
            ax.text(0.5, y_pos, learning_type, ha='center', va='center', 
                   fontsize=14, fontweight='bold')
            
            y_pos -= 0.12
            x_offset = 0.15
            
            for category, algorithms in categories.items():
                # Draw subcategory
                rect = patches.Rectangle((x_offset, y_pos-0.03), 0.25, 0.05, 
                                       linewidth=1, edgecolor='gray', 
                                       facecolor='white', alpha=0.8)
                ax.add_patch(rect)
                ax.text(x_offset + 0.125, y_pos, category, ha='center', va='center', 
                       fontsize=10, fontweight='bold')
                
                # List algorithms
                algo_text = '\n'.join([f"• {algo}" for algo in algorithms])
                ax.text(x_offset + 0.3, y_pos, algo_text, ha='left', va='center', 
                       fontsize=8)
                
                x_offset += 0.4
                if x_offset > 0.7:
                    x_offset = 0.15
                    y_pos -= 0.15
            
            y_pos -= 0.1
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Machine Learning Algorithms Taxonomy', fontsize=18, pad=20)
        
        plt.tight_layout()
        plt.show()
    
    def algorithm_characteristics(self):
        """Compare algorithm characteristics"""
        algorithms = [
            'Linear Regression', 'Logistic Regression', 'Decision Tree',
            'Random Forest', 'SVM', 'K-Means', 'Neural Network'
        ]
        
        characteristics = {
            'Algorithm': algorithms,
            'Learning Type': ['Supervised', 'Supervised', 'Supervised', 
                            'Supervised', 'Supervised', 'Unsupervised', 'Supervised'],
            'Problem Type': ['Regression', 'Classification', 'Both',
                           'Both', 'Both', 'Clustering', 'Both'],
            'Interpretability': ['High', 'High', 'High', 'Medium', 'Low', 'Medium', 'Low'],
            'Training Speed': ['Fast', 'Fast', 'Fast', 'Medium', 'Slow', 'Fast', 'Slow'],
            'Prediction Speed': ['Fast', 'Fast', 'Fast', 'Fast', 'Medium', 'Fast', 'Fast'],
            'Memory Usage': ['Low', 'Low', 'Low', 'Medium', 'Medium', 'Low', 'High'],
            'Handles Non-linear': ['No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes']
        }
        
        df = pd.DataFrame(characteristics)
        print("Algorithm Characteristics Comparison:")
        print(df.to_string(index=False))
        
        return df

# Demonstrate taxonomy
taxonomy = AlgorithmTaxonomy()
taxonomy.visualize_taxonomy()
characteristics_df = taxonomy.algorithm_characteristics()
```

## 2. Linear Algorithms {#linear-algorithms}

Linear algorithms assume a linear relationship between input features and the target variable.

### Linear Regression

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

class LinearAlgorithms:
    """Implementation and demonstration of linear algorithms"""
    
    def linear_regression_from_scratch(self, X, y):
        """Implement linear regression from scratch"""
        print("=== LINEAR REGRESSION FROM SCRATCH ===")
        
        # Add bias term (intercept)
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Normal equation: θ = (X^T X)^(-1) X^T y
        theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        
        print(f"Coefficients: {theta[1:]}")
        print(f"Intercept: {theta[0]}")
        
        # Make predictions
        y_pred = X_with_bias @ theta
        
        # Calculate metrics
        mse = np.mean((y - y_pred) ** 2)
        r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
        
        return theta, y_pred
    
    def gradient_descent_linear_regression(self, X, y, learning_rate=0.01, iterations=1000):
        """Implement linear regression using gradient descent"""
        print("\n=== LINEAR REGRESSION WITH GRADIENT DESCENT ===")
        
        # Initialize parameters
        m, n = X.shape
        theta = np.zeros(n + 1)  # +1 for intercept
        
        # Add bias term
        X_with_bias = np.column_stack([np.ones(m), X])
        
        # Store cost history
        cost_history = []
        
        for i in range(iterations):
            # Forward pass
            y_pred = X_with_bias @ theta
            
            # Calculate cost (MSE)
            cost = np.mean((y_pred - y) ** 2)
            cost_history.append(cost)
            
            # Calculate gradients
            gradients = (2/m) * X_with_bias.T @ (y_pred - y)
            
            # Update parameters
            theta -= learning_rate * gradients
            
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost:.6f}")
        
        print(f"Final coefficients: {theta[1:]}")
        print(f"Final intercept: {theta[0]}")
        
        # Plot cost history
        plt.figure(figsize=(10, 6))
        plt.plot(cost_history)
        plt.title('Cost Function During Training')
        plt.xlabel('Iteration')
        plt.ylabel('Cost (MSE)')
        plt.grid(True)
        plt.show()
        
        return theta, cost_history
    
    def regularized_regression_comparison(self, X, y):
        """Compare different regularization techniques"""
        print("\n=== REGULARIZED REGRESSION COMPARISON ===")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Different models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge (α=1.0)': Ridge(alpha=1.0),
            'Ridge (α=10.0)': Ridge(alpha=10.0),
            'Lasso (α=0.1)': Lasso(alpha=0.1),
            'Lasso (α=1.0)': Lasso(alpha=1.0)
        }
        
        results = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            results[name] = {
                'Train MSE': train_mse,
                'Test MSE': test_mse,
                'Train R²': train_r2,
                'Test R²': test_r2,
                'Coefficients': model.coef_,
                'Non-zero Coef': np.sum(np.abs(model.coef_) > 1e-6)
            }
        
        # Display results
        results_df = pd.DataFrame(results).T
        print(results_df[['Train MSE', 'Test MSE', 'Train R²', 'Test R²', 'Non-zero Coef']])
        
        # Visualize coefficients
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (name, model) in enumerate(models.items()):
            axes[i].bar(range(len(model.coef_)), model.coef_)
            axes[i].set_title(f'{name}\nCoefficients')
            axes[i].set_xlabel('Feature Index')
            axes[i].set_ylabel('Coefficient Value')
            axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.show()
        
        return results

# Generate sample data and demonstrate
np.random.seed(42)
X = np.random.randn(100, 5)
true_coef = np.array([1.5, -2.0, 0.5, 0.0, 3.0])
y = X @ true_coef + np.random.randn(100) * 0.1

linear_demo = LinearAlgorithms()
linear_demo.linear_regression_from_scratch(X, y)
linear_demo.gradient_descent_linear_regression(X, y)
linear_demo.regularized_regression_comparison(X, y)
```

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

class LogisticRegressionDemo:
    """Demonstrate logistic regression implementation and concepts"""
    
    def sigmoid_function(self):
        """Visualize the sigmoid function"""
        print("=== SIGMOID FUNCTION ===")
        
        z = np.linspace(-10, 10, 100)
        sigmoid = 1 / (1 + np.exp(-z))
        
        plt.figure(figsize=(10, 6))
        plt.plot(z, sigmoid, 'b-', linewidth=2, label='Sigmoid Function')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision Boundary')
        plt.axvline(x=0, color='g', linestyle='--', alpha=0.7)
        plt.xlabel('z (Linear Combination)')
        plt.ylabel('P(y=1|x)')
        plt.title('Sigmoid (Logistic) Function')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(-0.1, 1.1)
        plt.show()
        
        print("Key properties:")
        print("• Output range: (0, 1)")
        print("• Decision boundary at z = 0 (P = 0.5)")
        print("• Smooth, differentiable function")
        print("• Maps any real number to probability")
    
    def logistic_regression_from_scratch(self, X, y, learning_rate=0.01, iterations=1000):
        """Implement logistic regression from scratch"""
        print("\n=== LOGISTIC REGRESSION FROM SCRATCH ===")
        
        # Add bias term
        m, n = X.shape
        X_with_bias = np.column_stack([np.ones(m), X])
        theta = np.zeros(n + 1)
        
        def sigmoid(z):
            # Clip z to prevent overflow
            z = np.clip(z, -500, 500)
            return 1 / (1 + np.exp(-z))
        
        cost_history = []
        
        for i in range(iterations):
            # Forward pass
            z = X_with_bias @ theta
            predictions = sigmoid(z)
            
            # Calculate cost (log-likelihood)
            cost = -np.mean(y * np.log(predictions + 1e-15) + 
                           (1 - y) * np.log(1 - predictions + 1e-15))
            cost_history.append(cost)
            
            # Calculate gradients
            gradients = (1/m) * X_with_bias.T @ (predictions - y)
            
            # Update parameters
            theta -= learning_rate * gradients
            
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost:.6f}")
        
        print(f"Final coefficients: {theta[1:]}")
        print(f"Final intercept: {theta[0]}")
        
        # Plot cost history
        plt.figure(figsize=(10, 6))
        plt.plot(cost_history)
        plt.title('Cost Function During Training (Logistic Regression)')
        plt.xlabel('Iteration')
        plt.ylabel('Cost (Log-Likelihood)')
        plt.grid(True)
        plt.show()
        
        return theta, cost_history
    
    def binary_classification_example(self):
        """Complete binary classification example"""
        print("\n=== BINARY CLASSIFICATION EXAMPLE ===")
        
        # Generate sample data
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                                 n_informative=2, n_clusters_per_class=1, random_state=42)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train logistic regression
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Visualizations
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Data points and decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict_proba(mesh_points)[:, 1]
        Z = Z.reshape(xx.shape)
        
        axes[0].contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
        scatter = axes[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', edgecolors='black')
        axes[0].set_title('Decision Boundary')
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        plt.colorbar(scatter, ax=axes[0])
        
        # Plot 2: ROC Curve
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC Curve (AUC = {roc_auc:.2f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc="lower right")
        axes[1].grid(True)
        
        # Plot 3: Prediction probabilities
        axes[2].hist(y_pred_proba[y_test == 0], bins=20, alpha=0.7, label='Class 0', color='blue')
        axes[2].hist(y_pred_proba[y_test == 1], bins=20, alpha=0.7, label='Class 1', color='red')
        axes[2].axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
        axes[2].set_xlabel('Predicted Probability')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Prediction Probability Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return model, X_test, y_test, y_pred_proba

# Demonstrate logistic regression
logistic_demo = LogisticRegressionDemo()
logistic_demo.sigmoid_function()

# Generate sample data for scratch implementation
np.random.seed(42)
X_simple = np.random.randn(200, 2)
y_simple = (X_simple[:, 0] + X_simple[:, 1] > 0).astype(int)

logistic_demo.logistic_regression_from_scratch(X_simple, y_simple)
logistic_demo.binary_classification_example()
```

## 3. Tree-Based Algorithms {#tree-based-algorithms}

Tree-based algorithms create a model that predicts target values by learning simple decision rules inferred from data features.

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt

class TreeBasedAlgorithms:
    """Demonstrate tree-based algorithms"""
    
    def decision_tree_visualization(self):
        """Visualize how decision trees work"""
        print("=== DECISION TREE VISUALIZATION ===")
        
        # Create simple dataset
        from sklearn.datasets import load_iris
        iris = load_iris()
        
        # Use only two features for visualization
        X = iris.data[:, [0, 2]]  # sepal length, petal length
        y = iris.target
        
        # Train decision tree
        tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        tree.fit(X, y)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot 1: Decision tree structure
        plot_tree(tree, feature_names=['Sepal Length', 'Petal Length'], 
                 class_names=iris.target_names, filled=True, rounded=True, ax=axes[0])
        axes[0].set_title('Decision Tree Structure')
        
        # Plot 2: Decision boundaries
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        axes[1].contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
        scatter = axes[1].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
        axes[1].set_xlabel('Sepal Length')
        axes[1].set_ylabel('Petal Length')
        axes[1].set_title('Decision Boundaries')
        plt.colorbar(scatter, ax=axes[1])
        
        plt.tight_layout()
        plt.show()
        
        # Print tree rules
        print("\nSample Decision Rules:")
        print("• If petal length ≤ 2.45: Class = Setosa")
        print("• If petal length > 2.45 and ≤ 4.75: Class = Versicolor")
        print("• If petal length > 4.75: Class = Virginica")
        
        return tree
    
    def tree_depth_analysis(self):
        """Analyze effect of tree depth on performance"""
        print("\n=== TREE DEPTH ANALYSIS ===")
        
        # Generate sample data
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                                 n_redundant=2, random_state=42)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Test different depths
        depths = range(1, 21)
        train_scores = []
        test_scores = []
        
        for depth in depths:
            tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
            tree.fit(X_train, y_train)
            
            train_score = tree.score(X_train, y_train)
            test_score = tree.score(X_test, y_test)
            
            train_scores.append(train_score)
            test_scores.append(test_score)
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(depths, train_scores, 'o-', label='Training Accuracy', color='blue')
        plt.plot(depths, test_scores, 'o-', label='Testing Accuracy', color='red')
        plt.xlabel('Tree Depth')
        plt.ylabel('Accuracy')
        plt.title('Decision Tree Performance vs. Depth')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Find optimal depth
        optimal_depth = depths[np.argmax(test_scores)]
        print(f"Optimal tree depth: {optimal_depth}")
        print(f"Best test accuracy: {max(test_scores):.4f}")
        
        return depths, train_scores, test_scores
    
    def random_forest_demonstration(self):
        """Demonstrate Random Forest ensemble method"""
        print("\n=== RANDOM FOREST DEMONSTRATION ===")
        
        # Generate sample data
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                                 n_redundant=2, random_state=42)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Compare single tree vs random forest
        models = {
            'Single Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest (10 trees)': RandomForestClassifier(n_estimators=10, random_state=42),
            'Random Forest (50 trees)': RandomForestClassifier(n_estimators=50, random_state=42),
            'Random Forest (100 trees)': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            results[name] = {
                'Train Accuracy': train_score,
                'Test Accuracy': test_score,
                'Overfitting': train_score - test_score
            }
            
            if hasattr(model, 'feature_importances_'):
                results[name]['Feature Importance Std'] = np.std(model.feature_importances_)
        
        # Display results
        results_df = pd.DataFrame(results).T
        print("Model Comparison:")
        print(results_df)
        
        # Feature importance analysis
        rf_model = models['Random Forest (100 trees)']
        feature_importance = rf_model.feature_importances_
        
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Feature importance
        plt.subplot(1, 2, 1)
        plt.bar(range(len(feature_importance)), feature_importance)
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.title('Random Forest Feature Importance')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Number of trees vs performance
        n_trees = range(1, 101, 5)
        rf_scores = []
        
        for n in n_trees:
            rf = RandomForestClassifier(n_estimators=n, random_state=42)
            rf.fit(X_train, y_train)
            score = rf.score(X_test, y_test)
            rf_scores.append(score)
        
        plt.subplot(1, 2, 2)
        plt.plot(n_trees, rf_scores, 'o-', color='green')
        plt.xlabel('Number of Trees')
        plt.ylabel('Test Accuracy')
        plt.title('Random Forest Performance vs. Number of Trees')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    def tree_interpretability(self):
        """Demonstrate tree interpretability features"""
        print("\n=== TREE INTERPRETABILITY ===")
        
        # Use a simple dataset for clear interpretation
        from sklearn.datasets import load_wine
        wine = load_wine()
        
        # Select a few features for simplicity
        feature_indices = [0, 6, 9]  # alcohol, flavanoids, color_intensity
        X = wine.data[:, feature_indices]
        y = wine.target
        feature_names = [wine.feature_names[i] for i in feature_indices]
        
        # Train decision tree
        tree = DecisionTreeClassifier(max_depth=4, random_state=42)
        tree.fit(X, y)
        
        # Extract decision rules
        def get_rules(tree, feature_names, class_names):
            tree_ = tree.tree_
            feature_name = [
                feature_names[i] if i != -2
                else "undefined!"
                for i in tree_.feature
            ]
            
            def recurse(node, depth, parent_rule=""):
                indent = "  " * depth
                if tree_.feature[node] != -2:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                    left_rule = f"{parent_rule}({name} <= {threshold:.2f})"
                    right_rule = f"{parent_rule}({name} > {threshold:.2f})"
                    
                    print(f"{indent}if {name} <= {threshold:.2f}:")
                    recurse(tree_.children_left[node], depth + 1, left_rule)
                    print(f"{indent}else:  # if {name} > {threshold:.2f}")
                    recurse(tree_.children_right[node], depth + 1, right_rule)
                else:
                    value = tree_.value[node]
                    class_idx = np.argmax(value)
                    class_name = class_names[class_idx]
                    confidence = value[0][class_idx] / np.sum(value)
                    print(f"{indent}return {class_name} (confidence: {confidence:.2f})")
            
            recurse(0, 0)
        
        print("Decision Tree Rules:")
        get_rules(tree, feature_names, wine.target_names)
        
        # Feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': tree.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\nFeature Importance:")
        print(importance_df)
        
        return tree, importance_df

# Demonstrate tree-based algorithms
tree_demo = TreeBasedAlgorithms()
tree_demo.decision_tree_visualization()
tree_demo.tree_depth_analysis()
tree_demo.random_forest_demonstration()
tree_demo.tree_interpretability()
```

## 4. Instance-Based Algorithms {#instance-based-algorithms}

Instance-based algorithms make predictions based on similarity to stored training instances.

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import euclidean, manhattan, cosine

class InstanceBasedAlgorithms:
    """Demonstrate instance-based learning algorithms"""
    
    def knn_from_scratch(self, X_train, y_train, X_test, k=3):
        """Implement K-Nearest Neighbors from scratch"""
        print("=== K-NEAREST NEIGHBORS FROM SCRATCH ===")
        
        def euclidean_distance(point1, point2):
            return np.sqrt(np.sum((point1 - point2) ** 2))
        
        predictions = []
        
        for test_point in X_test:
            # Calculate distances to all training points
            distances = []
            for i, train_point in enumerate(X_train):
                dist = euclidean_distance(test_point, train_point)
                distances.append((dist, y_train[i]))
            
            # Sort by distance and get k nearest neighbors
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:k]
            
            # Make prediction (majority vote for classification)
            neighbor_labels = [label for _, label in k_nearest]
            prediction = max(set(neighbor_labels), key=neighbor_labels.count)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def distance_metrics_comparison(self):
        """Compare different distance metrics"""
        print("\n=== DISTANCE METRICS COMPARISON ===")
        
        # Generate sample data
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                                 n_informative=2, n_clusters_per_class=1, random_state=42)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Different distance metrics
        metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
        
        results = {}
        
        for metric in metrics:
            knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[metric] = accuracy
        
        print("Accuracy by Distance Metric:")
        for metric, acc in results.items():
            print(f"  {metric.capitalize()}: {acc:.4f}")
        
        # Visualize distance metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
            knn.fit(X_train, y_train)
            
            # Create decision boundary
            h = 0.02
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            axes[i].contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
            axes[i].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolors='black')
            axes[i].set_title(f'{metric.capitalize()} Distance\nAccuracy: {results[metric]:.3f}')
            axes[i].set_xlabel('Feature 1')
            axes[i].set_ylabel('Feature 2')
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    def k_value_analysis(self):
        """Analyze the effect of k value on KNN performance"""
        print("\n=== K VALUE ANALYSIS ===")
        
        # Generate sample data
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                                 n_redundant=2, random_state=42)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Test different k values
        k_values = range(1, 31)
        train_accuracies = []
        test_accuracies = []
        
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            
            train_acc = knn.score(X_train, y_train)
            test_acc = knn.score(X_test, y_test)
            
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, train_accuracies, 'o-', label='Training Accuracy', color='blue')
        plt.plot(k_values, test_accuracies, 'o-', label='Testing Accuracy', color='red')
        plt.xlabel('K Value')
        plt.ylabel('Accuracy')
        plt.title('KNN Performance vs. K Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Find optimal k
        optimal_k = k_values[np.argmax(test_accuracies)]
        best_accuracy = max(test_accuracies)
        
        print(f"Optimal K value: {optimal_k}")
        print(f"Best test accuracy: {best_accuracy:.4f}")
        
        print("\nKey Observations:")
        print("• Low K: High variance, low bias (overfitting)")
        print("• High K: Low variance, high bias (underfitting)")
        print("• Odd K values help avoid ties in binary classification")
        
        return k_values, train_accuracies, test_accuracies
    
    def knn_regression_example(self):
        """Demonstrate KNN for regression"""
        print("\n=== KNN REGRESSION EXAMPLE ===")
        
        # Generate sample regression data
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Compare different k values for regression
        k_values = [1, 3, 5, 10]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, k in enumerate(k_values):
            knn_reg = KNeighborsRegressor(n_neighbors=k)
            knn_reg.fit(X_train, y_train)
            
            # Create smooth prediction line
            X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_plot = knn_reg.predict(X_plot)
            
            # Plot
            axes[i].scatter(X_train, y_train, alpha=0.6, color='blue', label='Training Data')
            axes[i].scatter(X_test, y_test, alpha=0.6, color='red', label='Test Data')
            axes[i].plot(X_plot, y_plot, color='green', linewidth=2, label=f'KNN (k={k})')
            axes[i].set_title(f'KNN Regression (k={k})')
            axes[i].set_xlabel('Feature')
            axes[i].set_ylabel('Target')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate MSE for different k values
        mse_results = {}
        for k in k_values:
            knn_reg = KNeighborsRegressor(n_neighbors=k)
            knn_reg.fit(X_train, y_train)
            y_pred = knn_reg.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mse_results[k] = mse
        
        print("Mean Squared Error by K value:")
        for k, mse in mse_results.items():
            print(f"  K={k}: {mse:.2f}")
        
        return mse_results

# Demonstrate instance-based algorithms
instance_demo = InstanceBasedAlgorithms()

# Generate sample data for scratch implementation
np.random.seed(42)
X_simple = np.random.randn(50, 2)
y_simple = (X_simple[:, 0] + X_simple[:, 1] > 0).astype(int)
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple, y_simple, test_size=0.3, random_state=42)

# Test scratch implementation
scratch_predictions = instance_demo.knn_from_scratch(
    X_train_simple, y_train_simple, X_test_simple, k=3)
scratch_accuracy = accuracy_score(y_test_simple, scratch_predictions)
print(f"Scratch KNN Accuracy: {scratch_accuracy:.4f}")

instance_demo.distance_metrics_comparison()
instance_demo.k_value_analysis()
instance_demo.knn_regression_example()
```

## Summary

This chapter provided a comprehensive overview of major machine learning algorithm families:

1. **Algorithm Taxonomy**: Understanding how algorithms are categorized and their characteristics
2. **Linear Algorithms**: Simple yet powerful methods assuming linear relationships
3. **Tree-Based Algorithms**: Interpretable methods that create decision rules
4. **Instance-Based Algorithms**: Memory-based methods using similarity measures

### Key Takeaways:
- Different algorithms have different strengths and are suited for different types of problems
- Linear algorithms are fast and interpretable but limited to linear relationships
- Tree-based methods are highly interpretable and handle non-linear relationships well
- Instance-based methods are simple conceptually but can be computationally expensive
- Ensemble methods often outperform individual algorithms
- Understanding algorithm characteristics helps in selection and tuning

### Algorithm Selection Guidelines:
- **Small dataset**: Simple algorithms (linear models, KNN)
- **Large dataset**: Scalable algorithms (linear models, tree ensembles)
- **Interpretability required**: Decision trees, linear models
- **High accuracy needed**: Ensemble methods, neural networks
- **Non-linear relationships**: Tree-based methods, KNN, neural networks

---

## Exercises

1. **Implementation**: Implement a decision tree from scratch using the ID3 algorithm
2. **Comparison**: Compare all algorithms on the same dataset and analyze results
3. **Optimization**: Tune hyperparameters for each algorithm family
4. **Real-world Application**: Choose appropriate algorithms for a specific domain problem
5. **Ensemble Creation**: Create your own ensemble method combining different algorithm types

---

*Understanding the landscape of machine learning algorithms is crucial for becoming an effective practitioner. Each algorithm family has its place in the ML toolkit.* 