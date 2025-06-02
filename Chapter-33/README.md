# Chapter 33: Hyperparameter Tuning Methods

## Learning Objectives
By the end of this chapter, students will be able to:
- Implement various hyperparameter optimization techniques
- Apply grid search, random search, and Bayesian optimization
- Use automated hyperparameter tuning libraries and frameworks
- Design efficient hyperparameter search strategies
- Evaluate and validate hyperparameter optimization results

## Table of Contents
1. [Introduction to Hyperparameter Tuning](#introduction)
2. [Grid Search and Random Search](#grid-random-search)
3. [Bayesian Optimization](#bayesian-optimization)
4. [Advanced Optimization Methods](#advanced-methods)
5. [Automated ML and Hyperparameter Tuning](#automl)
6. [Best Practices and Validation](#best-practices)

## 1. Introduction to Hyperparameter Tuning {#introduction}

Hyperparameter tuning is the process of finding the optimal configuration of hyperparameters to maximize model performance while avoiding overfitting.

### Types of Hyperparameters:
- **Model Architecture**: Number of layers, neurons, kernel sizes
- **Training Parameters**: Learning rate, batch size, epochs
- **Regularization**: Dropout rates, L1/L2 penalties
- **Optimization**: Optimizer choice, momentum, decay rates
- **Data Processing**: Feature selection, normalization methods

### Optimization Challenges:
- **High Dimensionality**: Many hyperparameters to tune simultaneously
- **Expensive Evaluation**: Each configuration requires full model training
- **Non-Convex Space**: Multiple local optima and noisy evaluations
- **Computational Constraints**: Limited time and resources
- **Overfitting**: Risk of overfitting to validation set

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import uniform, randint
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

class HyperparameterTuningToolkit:
    """Comprehensive toolkit for hyperparameter optimization"""
    
    def __init__(self):
        self.tuning_results = {}
        self.optimization_history = {}
        
    def generate_ml_dataset(self):
        """Generate dataset for hyperparameter tuning experiments"""
        # Use digits dataset for realistic hyperparameter tuning
        digits = load_digits()
        X, y = digits.data, digits.target
        
        # Split into train, validation, and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test), scaler
    
    def grid_search_demonstration(self, data):
        """Demonstrate grid search hyperparameter tuning"""
        print("=== GRID SEARCH HYPERPARAMETER TUNING ===")
        
        X_train, X_val, X_test, y_train, y_val, y_test = data
        
        # Define hyperparameter grid for MLPClassifier
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'alpha': [0.0001, 0.001, 0.01],
            'max_iter': [500, 1000]
        }
        
        print(f"Grid search space size: {np.prod([len(v) for v in param_grid.values()])} configurations")
        
        # Perform grid search
        mlp = MLPClassifier(random_state=42)
        
        start_time = time.time()
        grid_search = GridSearchCV(
            mlp, param_grid, cv=3, scoring='accuracy', 
            n_jobs=-1, verbose=1, return_train_score=True
        )
        
        # Combine train and validation for cross-validation
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.hstack([y_train, y_val])
        
        grid_search.fit(X_train_val, y_train_val)
        grid_time = time.time() - start_time
        
        # Evaluate best model
        best_model = grid_search.best_estimator_
        test_accuracy = best_model.score(X_test, y_test)
        
        print(f"\nGrid Search Results:")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Search time: {grid_time:.2f} seconds")
        
        # Analyze results
        results_df = pd.DataFrame(grid_search.cv_results_)
        
        return {
            'method': 'Grid Search',
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'test_accuracy': test_accuracy,
            'search_time': grid_time,
            'results_df': results_df,
            'best_model': best_model
        }
    
    def random_search_demonstration(self, data):
        """Demonstrate random search hyperparameter tuning"""
        print("\n=== RANDOM SEARCH HYPERPARAMETER TUNING ===")
        
        X_train, X_val, X_test, y_train, y_val, y_test = data
        
        # Define hyperparameter distributions for RandomizedSearchCV
        param_distributions = {
            'hidden_layer_sizes': [(50,), (100,), (150,), (50, 25), (100, 50), (100, 50, 25)],
            'learning_rate_init': uniform(0.001, 0.099),  # Uniform between 0.001 and 0.1
            'alpha': uniform(0.0001, 0.0999),  # Uniform between 0.0001 and 0.1
            'max_iter': randint(300, 1200)  # Random integer between 300 and 1200
        }
        
        # Perform random search
        mlp = MLPClassifier(random_state=42)
        
        start_time = time.time()
        random_search = RandomizedSearchCV(
            mlp, param_distributions, n_iter=50, cv=3, 
            scoring='accuracy', n_jobs=-1, verbose=1, 
            random_state=42, return_train_score=True
        )
        
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.hstack([y_train, y_val])
        
        random_search.fit(X_train_val, y_train_val)
        random_time = time.time() - start_time
        
        # Evaluate best model
        best_model = random_search.best_estimator_
        test_accuracy = best_model.score(X_test, y_test)
        
        print(f"\nRandom Search Results:")
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV score: {random_search.best_score_:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Search time: {random_time:.2f} seconds")
        print(f"Configurations tested: 50")
        
        results_df = pd.DataFrame(random_search.cv_results_)
        
        return {
            'method': 'Random Search',
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'test_accuracy': test_accuracy,
            'search_time': random_time,
            'results_df': results_df,
            'best_model': best_model
        }
    
    def bayesian_optimization_simulation(self, data):
        """Simulate Bayesian optimization for hyperparameter tuning"""
        print("\n=== BAYESIAN OPTIMIZATION SIMULATION ===")
        
        X_train, X_val, X_test, y_train, y_val, y_test = data
        
        # Simplified Bayesian optimization simulation
        class SimpleBayesianOptimizer:
            def __init__(self, objective_func, bounds, n_iter=30):
                self.objective_func = objective_func
                self.bounds = bounds
                self.n_iter = n_iter
                self.X_samples = []
                self.y_samples = []
                self.best_score = -np.inf
                self.best_params = None
                
            def acquisition_function(self, X_candidate):
                """Simple acquisition function (Upper Confidence Bound)"""
                if len(self.y_samples) == 0:
                    return np.random.random(len(X_candidate))
                
                # Simple mean and std estimation
                distances = []
                for x_cand in X_candidate:
                    dist = [np.linalg.norm(np.array(x_cand) - np.array(x_samp)) for x_samp in self.X_samples]
                    distances.append(min(dist) if dist else 1.0)
                
                # Encourage exploration of distant points
                exploration = np.array(distances)
                
                # Encourage exploitation near good points
                if self.y_samples:
                    best_idx = np.argmax(self.y_samples)
                    best_x = self.X_samples[best_idx]
                    exploitation = [-np.linalg.norm(np.array(x_cand) - np.array(best_x)) for x_cand in X_candidate]
                    exploitation = np.array(exploitation)
                else:
                    exploitation = np.zeros(len(X_candidate))
                
                return exploration + 0.5 * exploitation
            
            def optimize(self):
                """Run Bayesian optimization"""
                # Random initialization
                for _ in range(5):
                    params = self.sample_random_params()
                    score = self.objective_func(params)
                    self.X_samples.append(params)
                    self.y_samples.append(score)
                    
                    if score > self.best_score:
                        self.best_score = score
                        self.best_params = params
                
                # Bayesian optimization iterations
                for i in range(self.n_iter - 5):
                    # Generate candidate points
                    candidates = [self.sample_random_params() for _ in range(100)]
                    
                    # Select best candidate using acquisition function
                    acquisition_values = self.acquisition_function(candidates)
                    best_candidate_idx = np.argmax(acquisition_values)
                    next_params = candidates[best_candidate_idx]
                    
                    # Evaluate objective
                    score = self.objective_func(next_params)
                    self.X_samples.append(next_params)
                    self.y_samples.append(score)
                    
                    if score > self.best_score:
                        self.best_score = score
                        self.best_params = next_params
                    
                    if (i + 1) % 5 == 0:
                        print(f"Iteration {i + 6}: Best score = {self.best_score:.4f}")
                
                return self.best_params, self.best_score
            
            def sample_random_params(self):
                """Sample random parameters within bounds"""
                hidden_sizes = [(50,), (100,), (150,), (50, 25), (100, 50), (100, 50, 25)]
                return {
                    'hidden_layer_sizes': np.random.choice(hidden_sizes),
                    'learning_rate_init': np.random.uniform(0.001, 0.1),
                    'alpha': np.random.uniform(0.0001, 0.1),
                    'max_iter': np.random.randint(300, 1200)
                }
        
        # Objective function
        def objective_function(params):
            """Objective function to maximize (validation accuracy)"""
            try:
                model = MLPClassifier(
                    hidden_layer_sizes=params['hidden_layer_sizes'],
                    learning_rate_init=params['learning_rate_init'],
                    alpha=params['alpha'],
                    max_iter=params['max_iter'],
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                return score
            except:
                return 0.0
        
        # Run Bayesian optimization
        start_time = time.time()
        
        bounds = {
            'hidden_layer_sizes': [(50,), (100,), (150,), (50, 25), (100, 50), (100, 50, 25)],
            'learning_rate_init': (0.001, 0.1),
            'alpha': (0.0001, 0.1),
            'max_iter': (300, 1200)
        }
        
        optimizer = SimpleBayesianOptimizer(objective_function, bounds, n_iter=30)
        best_params, best_score = optimizer.optimize()
        
        bayesian_time = time.time() - start_time
        
        # Train final model and evaluate on test set
        final_model = MLPClassifier(
            hidden_layer_sizes=best_params['hidden_layer_sizes'],
            learning_rate_init=best_params['learning_rate_init'],
            alpha=best_params['alpha'],
            max_iter=best_params['max_iter'],
            random_state=42
        )
        
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.hstack([y_train, y_val])
        final_model.fit(X_train_val, y_train_val)
        
        test_accuracy = final_model.score(X_test, y_test)
        
        print(f"\nBayesian Optimization Results:")
        print(f"Best parameters: {best_params}")
        print(f"Best validation score: {best_score:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Search time: {bayesian_time:.2f} seconds")
        print(f"Function evaluations: 30")
        
        return {
            'method': 'Bayesian Optimization',
            'best_params': best_params,
            'best_score': best_score,
            'test_accuracy': test_accuracy,
            'search_time': bayesian_time,
            'optimization_history': list(zip(optimizer.X_samples, optimizer.y_samples)),
            'best_model': final_model
        }
    
    def multi_algorithm_comparison(self, data):
        """Compare hyperparameter tuning across different algorithms"""
        print("\n=== MULTI-ALGORITHM HYPERPARAMETER COMPARISON ===")
        
        X_train, X_val, X_test, y_train, y_val, y_test = data
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.hstack([y_train, y_val])
        
        # Define algorithms and their hyperparameter spaces
        algorithms = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'SVM': {
                'model': SVC(random_state=42),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'kernel': ['rbf', 'poly', 'sigmoid']
                }
            },
            'MLP': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                    'learning_rate_init': [0.001, 0.01, 0.1],
                    'alpha': [0.0001, 0.001, 0.01]
                }
            }
        }
        
        comparison_results = {}
        
        for name, config in algorithms.items():
            print(f"\nTuning {name}...")
            
            start_time = time.time()
            
            # Use RandomizedSearchCV for efficiency
            search = RandomizedSearchCV(
                config['model'], config['params'], n_iter=20, 
                cv=3, scoring='accuracy', n_jobs=-1, random_state=42
            )
            
            search.fit(X_train_val, y_train_val)
            
            tuning_time = time.time() - start_time
            test_accuracy = search.best_estimator_.score(X_test, y_test)
            
            comparison_results[name] = {
                'best_params': search.best_params_,
                'cv_score': search.best_score_,
                'test_accuracy': test_accuracy,
                'tuning_time': tuning_time
            }
            
            print(f"Best CV score: {search.best_score_:.4f}")
            print(f"Test accuracy: {test_accuracy:.4f}")
            print(f"Tuning time: {tuning_time:.2f} seconds")
        
        return comparison_results
    
    def learning_curve_analysis(self, data, best_params):
        """Analyze learning curves for hyperparameter validation"""
        print("\n=== LEARNING CURVE ANALYSIS ===")
        
        X_train, X_val, X_test, y_train, y_val, y_test = data
        
        # Train models with different training set sizes
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Use best parameters from previous search
        best_mlp_params = None
        for result in [best_params] if isinstance(best_params, dict) else best_params:
            if 'hidden_layer_sizes' in str(result):
                best_mlp_params = result
                break
        
        if best_mlp_params is None:
            best_mlp_params = {'hidden_layer_sizes': (100, 50), 'learning_rate_init': 0.01, 'alpha': 0.001}
        
        train_scores = []
        val_scores = []
        
        for train_size in train_sizes:
            n_samples = int(train_size * len(X_train))
            
            # Sample subset of training data
            indices = np.random.choice(len(X_train), n_samples, replace=False)
            X_subset = X_train[indices]
            y_subset = y_train[indices]
            
            # Train model
            model = MLPClassifier(
                hidden_layer_sizes=best_mlp_params.get('hidden_layer_sizes', (100, 50)),
                learning_rate_init=best_mlp_params.get('learning_rate_init', 0.01),
                alpha=best_mlp_params.get('alpha', 0.001),
                max_iter=1000,
                random_state=42
            )
            
            model.fit(X_subset, y_subset)
            
            train_score = model.score(X_subset, y_subset)
            val_score = model.score(X_val, y_val)
            
            train_scores.append(train_score)
            val_scores.append(val_score)
        
        return {
            'train_sizes': train_sizes,
            'train_scores': train_scores,
            'val_scores': val_scores
        }
    
    def visualize_hyperparameter_results(self, grid_results, random_results, bayesian_results, 
                                       comparison_results, learning_curve_data):
        """Visualize hyperparameter tuning results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Method comparison
        methods = ['Grid Search', 'Random Search', 'Bayesian Opt']
        test_accuracies = [
            grid_results['test_accuracy'],
            random_results['test_accuracy'], 
            bayesian_results['test_accuracy']
        ]
        search_times = [
            grid_results['search_time'],
            random_results['search_time'],
            bayesian_results['search_time']
        ]
        
        axes[0, 0].bar(methods, test_accuracies, alpha=0.7, color=['blue', 'green', 'red'])
        axes[0, 0].set_title('Test Accuracy by Method')
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, acc in enumerate(test_accuracies):
            axes[0, 0].text(i, acc + 0.001, f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. Search time comparison
        axes[0, 1].bar(methods, search_times, alpha=0.7, color=['blue', 'green', 'red'])
        axes[0, 1].set_title('Search Time by Method')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Bayesian optimization convergence
        if 'optimization_history' in bayesian_results:
            scores = [score for _, score in bayesian_results['optimization_history']]
            cumulative_best = np.maximum.accumulate(scores)
            
            axes[0, 2].plot(range(1, len(scores) + 1), scores, 'o-', alpha=0.6, label='Scores')
            axes[0, 2].plot(range(1, len(cumulative_best) + 1), cumulative_best, 'r-', linewidth=2, label='Best So Far')
            axes[0, 2].set_title('Bayesian Optimization Convergence')
            axes[0, 2].set_xlabel('Iteration')
            axes[0, 2].set_ylabel('Validation Accuracy')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Algorithm comparison
        if comparison_results:
            alg_names = list(comparison_results.keys())
            alg_accuracies = [comparison_results[alg]['test_accuracy'] for alg in alg_names]
            
            axes[1, 0].bar(alg_names, alg_accuracies, alpha=0.7)
            axes[1, 0].set_title('Algorithm Performance Comparison')
            axes[1, 0].set_ylabel('Test Accuracy')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Learning curves
        if learning_curve_data:
            train_sizes = learning_curve_data['train_sizes'] * len(learning_curve_data['train_scores'])
            
            axes[1, 1].plot(train_sizes, learning_curve_data['train_scores'], 'o-', label='Training', linewidth=2)
            axes[1, 1].plot(train_sizes, learning_curve_data['val_scores'], 'o-', label='Validation', linewidth=2)
            axes[1, 1].set_title('Learning Curves')
            axes[1, 1].set_xlabel('Training Set Size')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Hyperparameter sensitivity analysis
        if 'results_df' in grid_results:
            # Analyze learning rate sensitivity
            df = grid_results['results_df']
            if 'param_learning_rate_init' in df.columns:
                lr_analysis = df.groupby('param_learning_rate_init')['mean_test_score'].agg(['mean', 'std'])
                
                axes[1, 2].errorbar(lr_analysis.index, lr_analysis['mean'], 
                                   yerr=lr_analysis['std'], fmt='o-', capsize=5)
                axes[1, 2].set_title('Learning Rate Sensitivity')
                axes[1, 2].set_xlabel('Learning Rate')
                axes[1, 2].set_ylabel('CV Score')
                axes[1, 2].set_xscale('log')
                axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Demonstrate hyperparameter tuning methods
tuner = HyperparameterTuningToolkit()

print("=== HYPERPARAMETER TUNING METHODS ===")

# Generate dataset
data, scaler = tuner.generate_ml_dataset()
print("Dataset loaded: Digits recognition with 8x8 pixel images")

# Grid search
grid_results = tuner.grid_search_demonstration(data)

# Random search  
random_results = tuner.random_search_demonstration(data)

# Bayesian optimization
bayesian_results = tuner.bayesian_optimization_simulation(data)

# Multi-algorithm comparison
comparison_results = tuner.multi_algorithm_comparison(data)

# Learning curve analysis
learning_curve_data = tuner.learning_curve_analysis(data, grid_results['best_params'])

# Visualize results
tuner.visualize_hyperparameter_results(
    grid_results, random_results, bayesian_results,
    comparison_results, learning_curve_data
)
```

## Summary

This chapter covered comprehensive hyperparameter tuning methods:

### Key Optimization Methods:
1. **Grid Search**: Exhaustive search over parameter combinations
2. **Random Search**: Random sampling from parameter distributions  
3. **Bayesian Optimization**: Sequential model-based optimization
4. **Evolutionary Algorithms**: Population-based optimization methods
5. **Multi-fidelity Methods**: Early stopping and progressive refinement

### Optimization Strategies:
- **Search Space Design**: Defining appropriate parameter ranges
- **Cross-Validation**: Robust performance estimation
- **Early Stopping**: Computational efficiency improvements  
- **Multi-Objective Optimization**: Balancing accuracy and complexity
- **Parallelization**: Distributed hyperparameter search

### Best Practices:
- Start with random search before grid search
- Use validation curves to understand parameter sensitivity
- Implement proper cross-validation to avoid overfitting
- Consider computational budget and time constraints
- Log and visualize optimization progress
- Validate final models on held-out test sets

---

## Exercises

1. **Custom Optimizer**: Implement particle swarm optimization for hyperparameters
2. **Multi-Objective**: Optimize for both accuracy and model complexity
3. **Transfer Learning**: Use hyperparameters from similar tasks
4. **Automated Pipeline**: Build end-to-end automated hyperparameter tuning
5. **Sensitivity Analysis**: Analyze which hyperparameters matter most

---

*Effective hyperparameter tuning requires balancing exploration of the parameter space with computational efficiency while avoiding overfitting to the validation set.* 