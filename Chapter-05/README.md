# Chapter 5: Classification Algorithms for Predictive Modeling

## Learning Objectives
- Understand classification vs regression problems
- Implement logistic regression and decision trees
- Apply proper evaluation metrics for classification
- Handle different types of classification challenges

## Table of Contents
1. [Introduction to Classification](#introduction)
2. [Logistic Regression](#logistic-regression)
3. [Decision Trees](#decision-trees)
4. [Model Evaluation](#evaluation)

## 1. Introduction to Classification {#introduction}

Classification predicts discrete categories or class labels, unlike regression which predicts continuous values.

### Key Concepts

**Binary Classification**: Two classes (spam/not spam, positive/negative)
**Multi-class Classification**: Multiple classes (animal species, product categories)
**Decision Boundaries**: Lines/surfaces that separate different classes

### Applications
- **Medical**: Disease diagnosis, treatment recommendation
- **Finance**: Credit approval, fraud detection
- **Technology**: Image recognition, sentiment analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Generate sample classification data
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1, random_state=42)

# Visualize the data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('Binary Classification Problem')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.show()

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")
```

## 2. Logistic Regression {#logistic-regression}

Logistic regression uses the sigmoid function to model the probability of class membership.

### Mathematical Foundation

**Linear Combination**: z = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ
**Sigmoid Function**: σ(z) = 1 / (1 + e^(-z))
**Probability**: P(y=1|x) = σ(z)

### Why Sigmoid?
- Maps any real number to (0,1) range
- Smooth, differentiable function
- Natural interpretation as probability

```python
class LogisticRegression:
    """Logistic regression implementation from scratch"""
    
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        # Prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Train the model using gradient descent"""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training loop
        for i in range(self.max_iterations):
            # Forward pass
            linear_pred = X @ self.weights + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Cost function (cross-entropy)
            cost = self.compute_cost(y, predictions)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1/n_samples) * X.T @ (predictions - y)
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if i % 200 == 0:
                print(f"Iteration {i}, Cost: {cost:.4f}")
    
    def compute_cost(self, y_true, y_pred):
        """Compute cross-entropy cost"""
        # Prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        linear_pred = X @ self.weights + self.bias
        return self.sigmoid(linear_pred)
    
    def predict(self, X):
        """Make binary predictions"""
        return (self.predict_proba(X) >= 0.5).astype(int)

# Demonstrate logistic regression
print("=== LOGISTIC REGRESSION ===")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
log_reg = LogisticRegression(learning_rate=0.1, max_iterations=1000)
log_reg.fit(X_train, y_train)

# Make predictions
train_pred = log_reg.predict(X_train)
test_pred = log_reg.predict(X_test)

# Evaluate
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"\nTraining Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Cost function
axes[0].plot(log_reg.cost_history)
axes[0].set_title('Training Cost')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Cost')
axes[0].grid(True)

# Decision boundary
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

mesh_points = np.c_[xx.ravel(), yy.ravel()]
Z = log_reg.predict_proba(mesh_points)
Z = Z.reshape(xx.shape)

axes[1].contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='viridis')
axes[1].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
axes[1].set_title('Decision Boundary')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

## 3. Decision Trees {#decision-trees}

Decision trees create hierarchical rules to classify data by recursively splitting the dataset.

### How It Works

1. **Start with root node** containing all data
2. **Find best split** that separates classes most effectively
3. **Create child nodes** based on the split
4. **Repeat recursively** until stopping criteria met
5. **Assign class labels** to leaf nodes

### Splitting Criteria

**Gini Impurity**: Measures how often a randomly chosen element would be incorrectly labeled
**Information Gain**: Reduction in entropy after a split

```python
class DecisionTreeNode:
    """Node in a decision tree"""
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.samples = 0

class DecisionTree:
    """Simple decision tree classifier"""
    
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def gini_impurity(self, y):
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def information_gain(self, parent, left_child, right_child):
        """Calculate information gain from a split"""
        n_parent = len(parent)
        n_left = len(left_child)
        n_right = len(right_child)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        parent_gini = self.gini_impurity(parent)
        left_gini = self.gini_impurity(left_child)
        right_gini = self.gini_impurity(right_child)
        
        weighted_avg = (n_left/n_parent) * left_gini + (n_right/n_parent) * right_gini
        return parent_gini - weighted_avg
    
    def best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                gain = self.information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        node = DecisionTreeNode()
        node.samples = len(y)
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            len(y) < self.min_samples_split or 
            len(np.unique(y)) == 1):
            # Leaf node
            node.value = np.bincount(y).argmax()
            return node
        
        # Find best split
        feature, threshold = self.best_split(X, y)
        
        if feature is None:
            # No good split found
            node.value = np.bincount(y).argmax()
            return node
        
        # Create split
        node.feature = feature
        node.threshold = threshold
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # Build subtrees
        node.left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def fit(self, X, y):
        """Train the decision tree"""
        self.root = self.build_tree(X, y)
    
    def predict_sample(self, sample, node):
        """Predict a single sample"""
        if node.value is not None:  # Leaf node
            return node.value
        
        if sample[node.feature] <= node.threshold:
            return self.predict_sample(sample, node.left)
        else:
            return self.predict_sample(sample, node.right)
    
    def predict(self, X):
        """Predict multiple samples"""
        return np.array([self.predict_sample(sample, self.root) for sample in X])

# Demonstrate decision tree
print("\n=== DECISION TREE ===")

# Train decision tree
dt = DecisionTree(max_depth=3, min_samples_split=5)
dt.fit(X_train, y_train)

# Make predictions
dt_train_pred = dt.predict(X_train)
dt_test_pred = dt.predict(X_test)

# Evaluate
dt_train_acc = accuracy_score(y_train, dt_train_pred)
dt_test_acc = accuracy_score(y_test, dt_test_pred)

print(f"Training Accuracy: {dt_train_acc:.4f}")
print(f"Test Accuracy: {dt_test_acc:.4f}")

# Visualize decision boundary
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Decision boundary
h = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
mesh_points = np.c_[xx.ravel(), yy.ravel()]
Z = dt.predict(mesh_points)
Z = Z.reshape(xx.shape)

axes[0].contourf(xx, yy, Z, alpha=0.6, cmap='viridis')
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
axes[0].set_title('Decision Tree Boundary')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# Confusion matrix
cm = confusion_matrix(y_test, dt_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()
```

## 4. Model Evaluation {#evaluation}

Proper evaluation is crucial for understanding classifier performance.

### Key Metrics

**Accuracy**: Correct predictions / Total predictions
**Precision**: True Positives / (True Positives + False Positives)
**Recall**: True Positives / (True Positives + False Negatives)
**F1-Score**: Harmonic mean of precision and recall

### When to Use Which Metric

- **Accuracy**: Balanced datasets, all classes equally important
- **Precision**: Cost of false positives is high (spam detection)
- **Recall**: Cost of false negatives is high (medical diagnosis)
- **F1-Score**: Balance between precision and recall

```python
def evaluate_classifier(y_true, y_pred, class_names=['Class 0', 'Class 1']):
    """Comprehensive evaluation of a classifier"""
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # For binary classification
    if len(np.unique(y_true)) == 2:
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
    else:
        print(f"Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return accuracy

# Compare models
print("=== MODEL COMPARISON ===")

print("Logistic Regression:")
log_reg_acc = evaluate_classifier(y_test, test_pred)

print("\nDecision Tree:")
dt_acc = evaluate_classifier(y_test, dt_test_pred)

print(f"\nSummary:")
print(f"Logistic Regression Test Accuracy: {log_reg_acc:.4f}")
print(f"Decision Tree Test Accuracy: {dt_acc:.4f}")

# Model comparison visualization
models = ['Logistic Regression', 'Decision Tree']
accuracies = [log_reg_acc, dt_acc]

plt.figure(figsize=(8, 6))
bars = plt.bar(models, accuracies, color=['blue', 'green'], alpha=0.7)
plt.title('Model Comparison')
plt.ylabel('Test Accuracy')
plt.ylim(0, 1)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom')

plt.show()
```

## Summary

This chapter covered fundamental classification algorithms:

### Key Concepts:
- **Classification** predicts discrete categories vs continuous values
- **Logistic Regression** uses sigmoid function for probabilistic classification
- **Decision Trees** create interpretable rule-based models
- **Evaluation** requires multiple metrics beyond simple accuracy

### Algorithm Comparison:

| Algorithm | Advantages | Disadvantages |
|-----------|------------|---------------|
| Logistic Regression | Probabilistic output, fast, linear boundary | Assumes linear separability |
| Decision Trees | Interpretable, handles non-linear data | Prone to overfitting, unstable |

### Best Practices:
- Choose evaluation metrics appropriate for your problem
- Consider class imbalance when interpreting results
- Use cross-validation for robust performance estimates
- Visualize decision boundaries when possible
- Compare multiple algorithms on your specific dataset

---

## Exercises

1. **Multi-class Extension**: Extend algorithms to handle multiple classes
2. **Feature Engineering**: Test different feature transformations
3. **Imbalanced Data**: Apply techniques for handling class imbalance
4. **Real Dataset**: Apply classification to a real-world dataset
5. **Ensemble Methods**: Combine multiple classifiers for better performance

---

*Classification is fundamental to many AI applications. Understanding these core algorithms provides the foundation for more advanced techniques.* 