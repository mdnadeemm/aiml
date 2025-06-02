# Chapter 32: Techniques for AI Model Optimization

## Learning Objectives
By the end of this chapter, students will be able to:
- Implement various model optimization techniques and algorithms
- Apply gradient descent variants and advanced optimizers
- Use regularization techniques to prevent overfitting
- Optimize neural network architectures and training procedures
- Implement learning rate scheduling and adaptive methods

## Table of Contents
1. [Introduction to Model Optimization](#introduction)
2. [Gradient Descent Variants](#gradient-descent)
3. [Advanced Optimization Algorithms](#advanced-optimizers)
4. [Regularization Techniques](#regularization)
5. [Learning Rate Optimization](#learning-rate)
6. [Architecture Optimization](#architecture)

## 1. Introduction to Model Optimization {#introduction}

Model optimization is the process of improving model performance by adjusting parameters, architecture, and training procedures to minimize loss functions and maximize accuracy.

### Key Optimization Challenges:
- **Local Minima**: Getting trapped in suboptimal solutions
- **Vanishing/Exploding Gradients**: Training instability in deep networks
- **Overfitting**: Models that don't generalize well
- **Computational Efficiency**: Training time and resource constraints
- **Hyperparameter Selection**: Finding optimal configuration

### Optimization Categories:
- **Parameter Optimization**: Finding optimal weights and biases
- **Hyperparameter Optimization**: Tuning learning rates, batch sizes, etc.
- **Architecture Optimization**: Designing optimal network structures
- **Training Optimization**: Improving training procedures and algorithms

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class ModelOptimizationToolkit:
    """Comprehensive toolkit for AI model optimization"""
    
    def __init__(self):
        self.optimization_results = {}
        self.learning_curves = {}
        
    def gradient_descent_variants_demo(self):
        """Demonstrate different gradient descent variants"""
        print("=== GRADIENT DESCENT VARIANTS DEMONSTRATION ===")
        
        # Generate synthetic data
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                                 n_redundant=5, n_clusters_per_class=1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Different solvers (gradient descent variants)
        solvers = {
            'SGD': 'sgd',
            'Adam': 'adam', 
            'L-BFGS': 'lbfgs'
        }
        
        results = {}
        
        for name, solver in solvers.items():
            print(f"\nTraining with {name} optimizer...")
            
            # Train model
            if solver == 'lbfgs':
                # L-BFGS works better with smaller networks
                model = MLPClassifier(hidden_layer_sizes=(50,), solver=solver, 
                                    max_iter=1000, random_state=42)
            else:
                model = MLPClassifier(hidden_layer_sizes=(100, 50), solver=solver,
                                    max_iter=1000, random_state=42)
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_accuracy = model.score(X_train_scaled, y_train)
            test_accuracy = model.score(X_test_scaled, y_test)
            
            results[name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'n_iter': model.n_iter_,
                'loss_curve': model.loss_curve_ if hasattr(model, 'loss_curve_') else None
            }
            
            print(f"Train Accuracy: {train_accuracy:.3f}")
            print(f"Test Accuracy: {test_accuracy:.3f}")
            print(f"Iterations: {model.n_iter_}")
        
        return results, (X_train_scaled, X_test_scaled, y_train, y_test)
    
    def implement_custom_optimizers(self):
        """Implement custom optimization algorithms from scratch"""
        print("\n=== CUSTOM OPTIMIZER IMPLEMENTATIONS ===")
        
        class CustomOptimizer:
            """Base class for custom optimizers"""
            def __init__(self, learning_rate=0.01):
                self.learning_rate = learning_rate
                self.iterations = 0
                
            def update(self, weights, gradients):
                raise NotImplementedError
        
        class SGDOptimizer(CustomOptimizer):
            """Stochastic Gradient Descent"""
            def update(self, weights, gradients):
                self.iterations += 1
                return weights - self.learning_rate * gradients
        
        class MomentumOptimizer(CustomOptimizer):
            """SGD with Momentum"""
            def __init__(self, learning_rate=0.01, momentum=0.9):
                super().__init__(learning_rate)
                self.momentum = momentum
                self.velocity = None
                
            def update(self, weights, gradients):
                if self.velocity is None:
                    self.velocity = np.zeros_like(weights)
                
                self.velocity = self.momentum * self.velocity - self.learning_rate * gradients
                self.iterations += 1
                return weights + self.velocity
        
        class AdamOptimizer(CustomOptimizer):
            """Adam Optimizer"""
            def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
                super().__init__(learning_rate)
                self.beta1 = beta1
                self.beta2 = beta2
                self.epsilon = epsilon
                self.m = None  # First moment
                self.v = None  # Second moment
                
            def update(self, weights, gradients):
                if self.m is None:
                    self.m = np.zeros_like(weights)
                    self.v = np.zeros_like(weights)
                
                self.iterations += 1
                
                # Update biased first moment estimate
                self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
                
                # Update biased second raw moment estimate
                self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
                
                # Compute bias-corrected first moment estimate
                m_corrected = self.m / (1 - self.beta1 ** self.iterations)
                
                # Compute bias-corrected second raw moment estimate
                v_corrected = self.v / (1 - self.beta2 ** self.iterations)
                
                # Update weights
                return weights - self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
        
        # Demonstrate with simple quadratic function
        def quadratic_function(x):
            """Simple quadratic function for optimization"""
            return x[0]**2 + x[1]**2 + 0.1*x[0]*x[1]
        
        def quadratic_gradient(x):
            """Gradient of quadratic function"""
            return np.array([2*x[0] + 0.1*x[1], 2*x[1] + 0.1*x[0]])
        
        # Test optimizers
        optimizers = {
            'SGD': SGDOptimizer(learning_rate=0.1),
            'Momentum': MomentumOptimizer(learning_rate=0.1, momentum=0.9),
            'Adam': AdamOptimizer(learning_rate=0.1)
        }
        
        results = {}
        
        for name, optimizer in optimizers.items():
            # Starting point
            x = np.array([5.0, 5.0])
            history = [x.copy()]
            function_values = [quadratic_function(x)]
            
            # Optimization loop
            for _ in range(100):
                grad = quadratic_gradient(x)
                x = optimizer.update(x, grad)
                history.append(x.copy())
                function_values.append(quadratic_function(x))
                
                # Stop if converged
                if np.linalg.norm(grad) < 1e-6:
                    break
            
            results[name] = {
                'history': np.array(history),
                'function_values': function_values,
                'final_point': x,
                'final_value': quadratic_function(x),
                'iterations': len(history) - 1
            }
            
            print(f"\n{name} Optimizer:")
            print(f"Final point: ({x[0]:.6f}, {x[1]:.6f})")
            print(f"Final value: {quadratic_function(x):.6f}")
            print(f"Iterations: {len(history) - 1}")
        
        return results
    
    def regularization_techniques_demo(self):
        """Demonstrate various regularization techniques"""
        print("\n=== REGULARIZATION TECHNIQUES ===")
        
        # Generate data prone to overfitting (high-dimensional, small sample)
        X, y = make_classification(n_samples=200, n_features=100, n_informative=20,
                                 n_redundant=80, n_clusters_per_class=1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Different regularization techniques
        regularization_configs = {
            'No Regularization': {'alpha': 0.0001, 'hidden_layer_sizes': (100, 50)},
            'L2 Regularization': {'alpha': 0.01, 'hidden_layer_sizes': (100, 50)},
            'Strong L2': {'alpha': 0.1, 'hidden_layer_sizes': (100, 50)},
            'Dropout (Early Stop)': {'alpha': 0.0001, 'hidden_layer_sizes': (100, 50), 'early_stopping': True},
            'Smaller Network': {'alpha': 0.0001, 'hidden_layer_sizes': (20, 10)}
        }
        
        regularization_results = {}
        
        for name, config in regularization_configs.items():
            print(f"\nTraining with {name}...")
            
            model = MLPClassifier(
                hidden_layer_sizes=config['hidden_layer_sizes'],
                alpha=config['alpha'],
                early_stopping=config.get('early_stopping', False),
                validation_fraction=0.2 if config.get('early_stopping', False) else 0.1,
                max_iter=1000,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            train_accuracy = model.score(X_train_scaled, y_train)
            test_accuracy = model.score(X_test_scaled, y_test)
            
            regularization_results[name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'overfitting': train_accuracy - test_accuracy
            }
            
            print(f"Train Accuracy: {train_accuracy:.3f}")
            print(f"Test Accuracy: {test_accuracy:.3f}")
            print(f"Overfitting Gap: {train_accuracy - test_accuracy:.3f}")
        
        return regularization_results
    
    def learning_rate_scheduling_demo(self):
        """Demonstrate learning rate scheduling techniques"""
        print("\n=== LEARNING RATE SCHEDULING ===")
        
        # Generate regression data
        X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Different learning rate strategies
        lr_strategies = {
            'Constant': {'learning_rate': 'constant', 'learning_rate_init': 0.001},
            'Invscaling': {'learning_rate': 'invscaling', 'learning_rate_init': 0.001, 'power_t': 0.5},
            'Adaptive': {'learning_rate': 'adaptive', 'learning_rate_init': 0.001}
        }
        
        lr_results = {}
        
        for name, config in lr_strategies.items():
            print(f"\nTraining with {name} learning rate...")
            
            model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                learning_rate=config['learning_rate'],
                learning_rate_init=config['learning_rate_init'],
                power_t=config.get('power_t', 0.5),
                max_iter=1000,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            lr_results[name] = {
                'model': model,
                'train_score': train_score,
                'test_score': test_score,
                'loss_curve': model.loss_curve_,
                'n_iter': model.n_iter_
            }
            
            print(f"Train R²: {train_score:.3f}")
            print(f"Test R²: {test_score:.3f}")
            print(f"Iterations: {model.n_iter_}")
        
        return lr_results
    
    def architecture_optimization_demo(self):
        """Demonstrate neural network architecture optimization"""
        print("\n=== ARCHITECTURE OPTIMIZATION ===")
        
        # Generate classification data
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                                 n_redundant=5, n_clusters_per_class=1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Different architectures
        architectures = {
            'Single Layer (Small)': (50,),
            'Single Layer (Large)': (200,),
            'Two Layers': (100, 50),
            'Three Layers': (100, 50, 25),
            'Deep Network': (100, 80, 60, 40, 20),
            'Wide Network': (200, 200)
        }
        
        architecture_results = {}
        
        for name, hidden_layers in architectures.items():
            print(f"\nTesting {name} architecture...")
            
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                alpha=0.001,
                max_iter=1000,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            train_accuracy = model.score(X_train_scaled, y_train)
            test_accuracy = model.score(X_test_scaled, y_test)
            
            # Calculate model complexity (number of parameters)
            n_params = 0
            layer_sizes = [X_train_scaled.shape[1]] + list(hidden_layers) + [len(np.unique(y_train))]
            
            for i in range(len(layer_sizes) - 1):
                n_params += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]  # weights + biases
            
            architecture_results[name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'n_parameters': n_params,
                'architecture': hidden_layers,
                'overfitting': train_accuracy - test_accuracy
            }
            
            print(f"Architecture: {hidden_layers}")
            print(f"Parameters: {n_params:,}")
            print(f"Train Accuracy: {train_accuracy:.3f}")
            print(f"Test Accuracy: {test_accuracy:.3f}")
            print(f"Overfitting: {train_accuracy - test_accuracy:.3f}")
        
        return architecture_results
    
    def model_ensemble_optimization(self):
        """Demonstrate model ensemble optimization techniques"""
        print("\n=== MODEL ENSEMBLE OPTIMIZATION ===")
        
        # Generate data
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                                 n_redundant=5, n_clusters_per_class=1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Individual models
        models = {
            'Neural Network 1': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42),
            'Neural Network 2': MLPClassifier(hidden_layer_sizes=(50, 25), random_state=43),
            'Neural Network 3': MLPClassifier(hidden_layer_sizes=(80, 40, 20), random_state=44),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        individual_results = {}
        predictions = {}
        
        print("Training individual models...")
        for name, model in models.items():
            if 'Neural Network' in name:
                model.fit(X_train_scaled, y_train)
                train_acc = model.score(X_train_scaled, y_train)
                test_acc = model.score(X_test_scaled, y_test)
                pred = model.predict(X_test_scaled)
                pred_proba = model.predict_proba(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                train_acc = model.score(X_train, y_train)
                test_acc = model.score(X_test, y_test)
                pred = model.predict(X_test)
                pred_proba = model.predict_proba(X_test)
            
            individual_results[name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc
            }
            
            predictions[name] = {
                'predictions': pred,
                'probabilities': pred_proba
            }
            
            print(f"{name}: Test Accuracy = {test_acc:.3f}")
        
        # Ensemble methods
        print("\nEnsemble Results:")
        
        # Voting ensemble
        vote_predictions = []
        for i in range(len(y_test)):
            votes = [predictions[name]['predictions'][i] for name in models.keys()]
            vote_predictions.append(max(set(votes), key=votes.count))
        
        voting_accuracy = accuracy_score(y_test, vote_predictions)
        print(f"Voting Ensemble: {voting_accuracy:.3f}")
        
        # Averaging probabilities
        avg_probabilities = np.mean([predictions[name]['probabilities'] for name in models.keys()], axis=0)
        avg_predictions = np.argmax(avg_probabilities, axis=1)
        averaging_accuracy = accuracy_score(y_test, avg_predictions)
        print(f"Probability Averaging: {averaging_accuracy:.3f}")
        
        ensemble_results = {
            'individual': individual_results,
            'voting_accuracy': voting_accuracy,
            'averaging_accuracy': averaging_accuracy
        }
        
        return ensemble_results
    
    def visualize_optimization_results(self, gd_results, custom_opt_results, lr_results, arch_results):
        """Visualize optimization results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Gradient Descent Variants Comparison
        methods = list(gd_results.keys())
        train_accs = [gd_results[method]['train_accuracy'] for method in methods]
        test_accs = [gd_results[method]['test_accuracy'] for method in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, train_accs, width, label='Train', alpha=0.7)
        axes[0, 0].bar(x + width/2, test_accs, width, label='Test', alpha=0.7)
        axes[0, 0].set_title('Gradient Descent Variants')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(methods)
        axes[0, 0].legend()
        
        # 2. Custom Optimizers Convergence
        for name, result in custom_opt_results.items():
            axes[0, 1].plot(result['function_values'], label=name, linewidth=2)
        
        axes[0, 1].set_title('Optimizer Convergence')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Function Value')
        axes[0, 1].set_yscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Learning Rate Strategies
        for name, result in lr_results.items():
            if result['loss_curve'] is not None:
                axes[0, 2].plot(result['loss_curve'], label=name, linewidth=2)
        
        axes[0, 2].set_title('Learning Rate Strategies')
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Architecture Complexity vs Performance
        arch_names = list(arch_results.keys())
        n_params = [arch_results[arch]['n_parameters'] for arch in arch_names]
        test_accs_arch = [arch_results[arch]['test_accuracy'] for arch in arch_names]
        
        scatter = axes[1, 0].scatter(n_params, test_accs_arch, s=100, alpha=0.7)
        axes[1, 0].set_title('Architecture Complexity vs Performance')
        axes[1, 0].set_xlabel('Number of Parameters')
        axes[1, 0].set_ylabel('Test Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add labels for points
        for i, arch in enumerate(arch_names):
            axes[1, 0].annotate(arch, (n_params[i], test_accs_arch[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 5. Overfitting Analysis
        train_accs_arch = [arch_results[arch]['train_accuracy'] for arch in arch_names]
        overfitting = [arch_results[arch]['overfitting'] for arch in arch_names]
        
        axes[1, 1].bar(range(len(arch_names)), overfitting, alpha=0.7)
        axes[1, 1].set_title('Overfitting by Architecture')
        axes[1, 1].set_xlabel('Architecture')
        axes[1, 1].set_ylabel('Train - Test Accuracy')
        axes[1, 1].set_xticks(range(len(arch_names)))
        axes[1, 1].set_xticklabels([arch.split('(')[0] for arch in arch_names], rotation=45)
        
        # 6. Optimizer Paths (2D visualization)
        x_range = np.linspace(-6, 6, 100)
        y_range = np.linspace(-6, 6, 100)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        Z = X_grid**2 + Y_grid**2 + 0.1*X_grid*Y_grid
        
        contour = axes[1, 2].contour(X_grid, Y_grid, Z, levels=20, alpha=0.5)
        
        colors = ['red', 'blue', 'green']
        for i, (name, result) in enumerate(custom_opt_results.items()):
            history = result['history']
            axes[1, 2].plot(history[:, 0], history[:, 1], 'o-', 
                           color=colors[i], label=name, linewidth=2, markersize=3)
        
        axes[1, 2].set_title('Optimizer Paths')
        axes[1, 2].set_xlabel('x1')
        axes[1, 2].set_ylabel('x2')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Demonstrate model optimization techniques
optimizer = ModelOptimizationToolkit()

print("=== TECHNIQUES FOR AI MODEL OPTIMIZATION ===")

# Gradient descent variants
gd_results, test_data = optimizer.gradient_descent_variants_demo()

# Custom optimizers
custom_opt_results = optimizer.implement_custom_optimizers()

# Regularization techniques
reg_results = optimizer.regularization_techniques_demo()

# Learning rate scheduling
lr_results = optimizer.learning_rate_scheduling_demo()

# Architecture optimization
arch_results = optimizer.architecture_optimization_demo()

# Model ensemble optimization
ensemble_results = optimizer.model_ensemble_optimization()

# Visualize results
optimizer.visualize_optimization_results(gd_results, custom_opt_results, lr_results, arch_results)
```

## Summary

This chapter covered comprehensive techniques for AI model optimization:

### Key Optimization Techniques:
1. **Gradient Descent Variants**: SGD, Adam, L-BFGS with different convergence properties
2. **Advanced Optimizers**: Momentum, RMSprop, Adam with adaptive learning rates
3. **Regularization**: L1/L2 regularization, dropout, early stopping
4. **Learning Rate Scheduling**: Constant, adaptive, and decay strategies
5. **Architecture Optimization**: Network depth, width, and structure design

### Optimization Strategies:
- **Parameter Optimization**: Finding optimal weights through gradient-based methods
- **Hyperparameter Tuning**: Systematic search for optimal configuration
- **Regularization**: Preventing overfitting through various constraints
- **Ensemble Methods**: Combining multiple models for better performance

### Best Practices:
- Monitor both training and validation performance
- Use appropriate regularization for dataset size
- Implement learning rate scheduling for better convergence
- Consider model complexity vs. performance trade-offs
- Apply ensemble methods for robust predictions
- Validate optimization choices with cross-validation

---

## Exercises

1. **Custom Optimizer**: Implement RMSprop optimizer from scratch
2. **Regularization Comparison**: Compare different regularization techniques
3. **Architecture Search**: Implement neural architecture search algorithm
4. **Learning Schedule**: Design adaptive learning rate schedule
5. **Ensemble Optimization**: Create weighted ensemble with optimized weights

---

*Effective model optimization requires understanding the trade-offs between performance, complexity, and computational efficiency while preventing overfitting through appropriate regularization.* 