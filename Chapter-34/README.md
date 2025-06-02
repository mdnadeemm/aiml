# Chapter 34: Model Evaluation and Validation

## Learning Objectives
By the end of this chapter, students will be able to:
- Implement comprehensive model evaluation frameworks
- Apply various validation techniques and cross-validation methods
- Use appropriate metrics for different types of ML problems
- Detect and diagnose overfitting and underfitting
- Design robust evaluation pipelines for production systems

## Table of Contents
1. [Introduction to Model Evaluation](#introduction)
2. [Cross-Validation Techniques](#cross-validation)
3. [Evaluation Metrics](#metrics)
4. [Bias-Variance Analysis](#bias-variance)
5. [Model Diagnostics](#diagnostics)
6. [Production Evaluation](#production)

## 1. Introduction to Model Evaluation {#introduction}

Model evaluation is the process of assessing how well a machine learning model performs on unseen data and determining its readiness for deployment.

### Evaluation Objectives:
- **Performance Assessment**: Measure model accuracy and reliability
- **Generalization**: Estimate performance on unseen data
- **Model Selection**: Compare different algorithms and configurations
- **Deployment Readiness**: Validate production suitability
- **Continuous Monitoring**: Track performance over time

### Key Challenges:
- **Data Leakage**: Information from future leaking into training
- **Selection Bias**: Non-representative test sets
- **Temporal Dependencies**: Time-series specific considerations
- **Distribution Shift**: Changes in data patterns over time
- **Multiple Objectives**: Balancing accuracy, fairness, and efficiency

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification, make_regression, load_breast_cancer
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                   validation_curve, learning_curve, KFold,
                                   StratifiedKFold, TimeSeriesSplit)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_auc_score,
                           roc_curve, precision_recall_curve, mean_squared_error,
                           mean_absolute_error, r2_score)
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluationFramework:
    """Comprehensive framework for model evaluation and validation"""
    
    def __init__(self):
        self.evaluation_results = {}
        self.validation_curves = {}
        
    def create_evaluation_datasets(self):
        """Create datasets for evaluation demonstrations"""
        
        # Classification dataset
        X_class, y_class = make_classification(
            n_samples=1000, n_features=20, n_informative=15,
            n_redundant=5, n_clusters_per_class=1, random_state=42
        )
        
        # Regression dataset
        X_reg, y_reg = make_regression(
            n_samples=1000, n_features=20, noise=0.1, random_state=42
        )
        
        # Real-world dataset
        cancer = load_breast_cancer()
        X_real, y_real = cancer.data, cancer.target
        
        return {
            'classification': (X_class, y_class),
            'regression': (X_reg, y_reg),
            'real_world': (X_real, y_real)
        }
    
    def comprehensive_cross_validation(self, X, y, task_type='classification'):
        """Demonstrate various cross-validation techniques"""
        print("=== COMPREHENSIVE CROSS-VALIDATION ===")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Define models
        if task_type == 'classification':
            models = {
                'Logistic Regression': LogisticRegression(random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
            }
            scoring = 'accuracy'
        else:
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
            }
            scoring = 'neg_mean_squared_error'
        
        # Cross-validation strategies
        cv_strategies = {
            'K-Fold (5)': KFold(n_splits=5, shuffle=True, random_state=42),
            'K-Fold (10)': KFold(n_splits=10, shuffle=True, random_state=42),
            'Stratified (5)': StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if task_type == 'classification' else None,
            'Time Series': TimeSeriesSplit(n_splits=5)
        }
        
        # Remove None values
        cv_strategies = {k: v for k, v in cv_strategies.items() if v is not None}
        
        cv_results = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}:")
            model_results = {}
            
            for cv_name, cv_strategy in cv_strategies.items():
                if model_name == 'Neural Network' and task_type == 'classification':
                    scores = cross_val_score(model, X_scaled, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)
                else:
                    X_to_use = X_scaled if 'Regression' in model_name or 'Neural' in model_name else X
                    scores = cross_val_score(model, X_to_use, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)
                
                model_results[cv_name] = {
                    'scores': scores,
                    'mean': scores.mean(),
                    'std': scores.std()
                }
                
                print(f"  {cv_name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            cv_results[model_name] = model_results
        
        return cv_results
    
    def detailed_classification_evaluation(self, X, y):
        """Comprehensive classification model evaluation"""
        print("\n=== DETAILED CLASSIFICATION EVALUATION ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            
            # Train model
            if name in ['Logistic Regression', 'SVM']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'auc': roc_auc_score(y_test, y_prob)
            }
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # ROC curve
            fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
            
            # Precision-Recall curve
            precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
            
            results[name] = {
                'metrics': metrics,
                'confusion_matrix': cm,
                'roc_curve': (fpr, tpr, roc_thresholds),
                'pr_curve': (precision, recall, pr_thresholds),
                'predictions': y_pred,
                'probabilities': y_prob
            }
            
            # Print metrics
            for metric, value in metrics.items():
                print(f"  {metric.capitalize()}: {value:.4f}")
        
        return results, (X_test, y_test)
    
    def bias_variance_analysis(self, X, y):
        """Analyze bias-variance tradeoff"""
        print("\n=== BIAS-VARIANCE ANALYSIS ===")
        
        # Test different model complexities
        max_depths = [1, 2, 5, 10, 15, 20, None]
        n_estimators_list = [10, 50, 100, 200, 500]
        
        # Bias-variance for Random Forest depth
        depth_results = []
        
        for depth in max_depths:
            bootstrap_scores = []
            
            # Bootstrap sampling for bias-variance estimation
            for i in range(50):
                # Bootstrap sample
                indices = np.random.choice(len(X), len(X), replace=True)
                X_boot = X[indices]
                y_boot = y[indices]
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_boot, y_boot, test_size=0.3, random_state=i
                )
                
                # Train model
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=depth, random_state=42
                )
                model.fit(X_train, y_train)
                
                # Evaluate
                score = model.score(X_test, y_test)
                bootstrap_scores.append(score)
            
            depth_results.append({
                'depth': depth if depth is not None else 'None',
                'scores': bootstrap_scores,
                'mean': np.mean(bootstrap_scores),
                'variance': np.var(bootstrap_scores),
                'bias_estimate': 1 - np.mean(bootstrap_scores)  # Simplified bias estimate
            })
            
            print(f"Max Depth {depth}: Mean={np.mean(bootstrap_scores):.4f}, "
                  f"Var={np.var(bootstrap_scores):.6f}")
        
        return depth_results
    
    def learning_curves_analysis(self, X, y):
        """Generate and analyze learning curves"""
        print("\n=== LEARNING CURVES ANALYSIS ===")
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42)
        }
        
        learning_results = {}
        
        for name, model in models.items():
            print(f"\nGenerating learning curves for {name}...")
            
            # Prepare data
            X_to_use = X
            if name == 'Logistic Regression':
                scaler = StandardScaler()
                X_to_use = scaler.fit_transform(X)
            
            # Generate learning curves
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_to_use, y, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='accuracy', random_state=42
            )
            
            learning_results[name] = {
                'train_sizes': train_sizes,
                'train_scores': train_scores,
                'val_scores': val_scores,
                'train_mean': train_scores.mean(axis=1),
                'train_std': train_scores.std(axis=1),
                'val_mean': val_scores.mean(axis=1),
                'val_std': val_scores.std(axis=1)
            }
            
            print(f"  Final training score: {train_scores.mean(axis=1)[-1]:.4f}")
            print(f"  Final validation score: {val_scores.mean(axis=1)[-1]:.4f}")
            print(f"  Overfitting gap: {train_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-1]:.4f}")
        
        return learning_results
    
    def validation_curves_analysis(self, X, y):
        """Generate validation curves for hyperparameter analysis"""
        print("\n=== VALIDATION CURVES ANALYSIS ===")
        
        # Random Forest: n_estimators
        rf_param_range = [10, 20, 50, 100, 200, 500]
        
        train_scores, val_scores = validation_curve(
            RandomForestClassifier(random_state=42), X, y,
            param_name='n_estimators', param_range=rf_param_range,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        rf_results = {
            'param_range': rf_param_range,
            'train_scores': train_scores,
            'val_scores': val_scores,
            'train_mean': train_scores.mean(axis=1),
            'train_std': train_scores.std(axis=1),
            'val_mean': val_scores.mean(axis=1),
            'val_std': val_scores.std(axis=1)
        }
        
        # Logistic Regression: C parameter
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lr_param_range = [0.001, 0.01, 0.1, 1, 10, 100]
        
        train_scores_lr, val_scores_lr = validation_curve(
            LogisticRegression(random_state=42), X_scaled, y,
            param_name='C', param_range=lr_param_range,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        lr_results = {
            'param_range': lr_param_range,
            'train_scores': train_scores_lr,
            'val_scores': val_scores_lr,
            'train_mean': train_scores_lr.mean(axis=1),
            'train_std': train_scores_lr.std(axis=1),
            'val_mean': val_scores_lr.mean(axis=1),
            'val_std': val_scores_lr.std(axis=1)
        }
        
        print("Random Forest n_estimators analysis:")
        for i, n_est in enumerate(rf_param_range):
            print(f"  {n_est}: Train={rf_results['train_mean'][i]:.4f}, "
                  f"Val={rf_results['val_mean'][i]:.4f}")
        
        print("\nLogistic Regression C parameter analysis:")
        for i, c in enumerate(lr_param_range):
            print(f"  {c}: Train={lr_results['train_mean'][i]:.4f}, "
                  f"Val={lr_results['val_mean'][i]:.4f}")
        
        return {'Random Forest': rf_results, 'Logistic Regression': lr_results}
    
    def model_diagnostic_tests(self, X, y):
        """Comprehensive model diagnostic tests"""
        print("\n=== MODEL DIAGNOSTIC TESTS ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train baseline model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 1. Overfitting test
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        overfitting_gap = train_score - test_score
        
        print(f"Overfitting Analysis:")
        print(f"  Training Score: {train_score:.4f}")
        print(f"  Test Score: {test_score:.4f}")
        print(f"  Overfitting Gap: {overfitting_gap:.4f}")
        
        if overfitting_gap > 0.1:
            print("  ⚠️  Model shows signs of overfitting")
        else:
            print("  ✅ Model appears to generalize well")
        
        # 2. Feature importance stability
        feature_importances = []
        for i in range(10):
            # Bootstrap sample
            indices = np.random.choice(len(X_train), len(X_train), replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]
            
            # Train model
            boot_model = RandomForestClassifier(n_estimators=100, random_state=i)
            boot_model.fit(X_boot, y_boot)
            feature_importances.append(boot_model.feature_importances_)
        
        feature_importances = np.array(feature_importances)
        importance_mean = feature_importances.mean(axis=0)
        importance_std = feature_importances.std(axis=0)
        
        # Coefficient of variation for feature importance stability
        cv_importance = importance_std / (importance_mean + 1e-8)
        stable_features = np.sum(cv_importance < 0.5)
        
        print(f"\nFeature Importance Stability:")
        print(f"  Stable features (CV < 0.5): {stable_features}/{len(importance_mean)}")
        print(f"  Most stable feature: {np.argmin(cv_importance)} (CV: {cv_importance.min():.3f})")
        print(f"  Least stable feature: {np.argmax(cv_importance)} (CV: {cv_importance.max():.3f})")
        
        # 3. Prediction confidence analysis
        y_prob = model.predict_proba(X_test)
        max_probs = np.max(y_prob, axis=1)
        
        high_confidence = np.sum(max_probs > 0.8) / len(max_probs)
        low_confidence = np.sum(max_probs < 0.6) / len(max_probs)
        
        print(f"\nPrediction Confidence Analysis:")
        print(f"  High confidence predictions (>0.8): {high_confidence:.2%}")
        print(f"  Low confidence predictions (<0.6): {low_confidence:.2%}")
        print(f"  Average prediction confidence: {max_probs.mean():.3f}")
        
        return {
            'overfitting': {'train_score': train_score, 'test_score': test_score, 'gap': overfitting_gap},
            'feature_stability': {'mean': importance_mean, 'std': importance_std, 'cv': cv_importance},
            'prediction_confidence': {'probs': max_probs, 'high_conf': high_confidence, 'low_conf': low_confidence}
        }
    
    def visualize_evaluation_results(self, classification_results, learning_results, 
                                   validation_results, bias_variance_results):
        """Comprehensive visualization of evaluation results"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # 1. Classification metrics comparison
        if classification_results:
            models = list(classification_results.keys())
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
            
            metric_data = []
            for model in models:
                for metric in metrics:
                    metric_data.append({
                        'Model': model,
                        'Metric': metric.capitalize(),
                        'Score': classification_results[model]['metrics'][metric]
                    })
            
            metric_df = pd.DataFrame(metric_data)
            
            sns.barplot(data=metric_df, x='Metric', y='Score', hue='Model', ax=axes[0, 0])
            axes[0, 0].set_title('Classification Metrics Comparison')
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. ROC Curves
        if classification_results:
            for model_name, results in classification_results.items():
                fpr, tpr, _ = results['roc_curve']
                auc = results['metrics']['auc']
                axes[0, 1].plot(fpr, tpr, label=f'{model_name} (AUC: {auc:.3f})', linewidth=2)
            
            axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curves')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curves
        if classification_results:
            for model_name, results in classification_results.items():
                precision, recall, _ = results['pr_curve']
                axes[0, 2].plot(recall, precision, label=model_name, linewidth=2)
            
            axes[0, 2].set_xlabel('Recall')
            axes[0, 2].set_ylabel('Precision')
            axes[0, 2].set_title('Precision-Recall Curves')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Learning Curves
        if learning_results:
            for model_name, results in learning_results.items():
                train_sizes = results['train_sizes']
                train_mean = results['train_mean']
                train_std = results['train_std']
                val_mean = results['val_mean']
                val_std = results['val_std']
                
                axes[1, 0].plot(train_sizes, train_mean, 'o-', label=f'{model_name} Train', linewidth=2)
                axes[1, 0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
                
                axes[1, 0].plot(train_sizes, val_mean, 's--', label=f'{model_name} Val', linewidth=2)
                axes[1, 0].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
            
            axes[1, 0].set_xlabel('Training Set Size')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title('Learning Curves')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Validation Curves - Random Forest
        if validation_results and 'Random Forest' in validation_results:
            rf_results = validation_results['Random Forest']
            param_range = rf_results['param_range']
            train_mean = rf_results['train_mean']
            train_std = rf_results['train_std']
            val_mean = rf_results['val_mean']
            val_std = rf_results['val_std']
            
            axes[1, 1].plot(param_range, train_mean, 'o-', label='Training', linewidth=2)
            axes[1, 1].fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
            
            axes[1, 1].plot(param_range, val_mean, 's-', label='Validation', linewidth=2)
            axes[1, 1].fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1)
            
            axes[1, 1].set_xlabel('n_estimators')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Validation Curve - Random Forest')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Validation Curves - Logistic Regression
        if validation_results and 'Logistic Regression' in validation_results:
            lr_results = validation_results['Logistic Regression']
            param_range = lr_results['param_range']
            train_mean = lr_results['train_mean']
            train_std = lr_results['train_std']
            val_mean = lr_results['val_mean']
            val_std = lr_results['val_std']
            
            axes[1, 2].semilogx(param_range, train_mean, 'o-', label='Training', linewidth=2)
            axes[1, 2].fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
            
            axes[1, 2].semilogx(param_range, val_mean, 's-', label='Validation', linewidth=2)
            axes[1, 2].fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1)
            
            axes[1, 2].set_xlabel('C (Regularization)')
            axes[1, 2].set_ylabel('Score')
            axes[1, 2].set_title('Validation Curve - Logistic Regression')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Bias-Variance Analysis
        if bias_variance_results:
            depths = [str(r['depth']) for r in bias_variance_results]
            means = [r['mean'] for r in bias_variance_results]
            variances = [r['variance'] for r in bias_variance_results]
            
            axes[2, 0].plot(depths, means, 'o-', label='Mean Score', linewidth=2)
            axes[2, 0].set_xlabel('Max Depth')
            axes[2, 0].set_ylabel('Score')
            axes[2, 0].set_title('Bias Analysis (Mean Score)')
            axes[2, 0].tick_params(axis='x', rotation=45)
            axes[2, 0].grid(True, alpha=0.3)
            
            axes[2, 1].plot(depths, variances, 's-', color='red', label='Variance', linewidth=2)
            axes[2, 1].set_xlabel('Max Depth')
            axes[2, 1].set_ylabel('Variance')
            axes[2, 1].set_title('Variance Analysis')
            axes[2, 1].tick_params(axis='x', rotation=45)
            axes[2, 1].grid(True, alpha=0.3)
        
        # 8. Confusion Matrix (first model)
        if classification_results:
            first_model = list(classification_results.keys())[0]
            cm = classification_results[first_model]['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2, 2])
            axes[2, 2].set_xlabel('Predicted')
            axes[2, 2].set_ylabel('Actual')
            axes[2, 2].set_title(f'Confusion Matrix - {first_model}')
        
        plt.tight_layout()
        plt.show()

# Demonstrate model evaluation framework
evaluator = ModelEvaluationFramework()

print("=== MODEL EVALUATION AND VALIDATION ===")

# Create datasets
datasets = evaluator.create_evaluation_datasets()
print("Created evaluation datasets: classification, regression, and real-world")

# Use real-world dataset for comprehensive evaluation
X_real, y_real = datasets['real_world']

# Cross-validation analysis
cv_results = evaluator.comprehensive_cross_validation(X_real, y_real, 'classification')

# Detailed classification evaluation
classification_results, test_data = evaluator.detailed_classification_evaluation(X_real, y_real)

# Bias-variance analysis
bias_variance_results = evaluator.bias_variance_analysis(X_real, y_real)

# Learning curves
learning_results = evaluator.learning_curves_analysis(X_real, y_real)

# Validation curves
validation_results = evaluator.validation_curves_analysis(X_real, y_real)

# Model diagnostics
diagnostic_results = evaluator.model_diagnostic_tests(X_real, y_real)

# Visualize results
evaluator.visualize_evaluation_results(
    classification_results, learning_results, validation_results, bias_variance_results
)
```

## Summary

This chapter provided comprehensive model evaluation and validation techniques:

### Key Evaluation Methods:
1. **Cross-Validation**: K-fold, stratified, time-series splitting strategies
2. **Performance Metrics**: Accuracy, precision, recall, F1-score, AUC-ROC
3. **Learning Curves**: Training set size vs. performance analysis
4. **Validation Curves**: Hyperparameter sensitivity analysis
5. **Bias-Variance Analysis**: Understanding model complexity trade-offs

### Evaluation Best Practices:
- Use appropriate metrics for the problem type and business objectives
- Implement proper cross-validation to get robust performance estimates
- Analyze learning curves to detect overfitting and underfitting
- Validate on truly unseen data to estimate production performance
- Monitor model performance continuously after deployment

### Diagnostic Techniques:
- **Overfitting Detection**: Compare training vs. validation performance
- **Feature Stability**: Bootstrap analysis of feature importance
- **Prediction Confidence**: Analyze model uncertainty
- **Distribution Shift**: Monitor data changes over time
- **Fairness Assessment**: Evaluate performance across subgroups

---

## Exercises

1. **Custom Metrics**: Implement domain-specific evaluation metrics
2. **Temporal Validation**: Design evaluation for time-series data
3. **Multi-Objective Evaluation**: Balance accuracy, fairness, and efficiency
4. **Production Monitoring**: Build continuous evaluation pipeline
5. **A/B Testing**: Design experiment framework for model comparison

---

*Robust model evaluation ensures reliable performance estimates and helps build trustworthy AI systems that generalize well to real-world scenarios.* 