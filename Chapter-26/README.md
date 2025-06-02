# Chapter 26: Fairness in AI Applications

## Learning Objectives
By the end of this chapter, students will be able to:
- Define and differentiate various concepts of fairness in AI
- Implement fairness metrics and constraints in AI systems
- Apply fairness-aware machine learning techniques
- Design fair AI systems for specific application domains
- Evaluate trade-offs between fairness, accuracy, and other objectives

## Table of Contents
1. [Concepts of Fairness](#concepts)
2. [Fairness Metrics](#metrics)
3. [Fairness-Aware ML](#fairness-aware-ml)
4. [Implementation Strategies](#implementation)
5. [Domain-Specific Considerations](#domain-specific)
6. [Trade-offs and Limitations](#trade-offs)

## 1. Concepts of Fairness {#concepts}

Fairness in AI is multifaceted, with different definitions appropriate for different contexts and applications.

### Individual Fairness
Similar individuals should receive similar outcomes
- **Definition**: Treat similar people similarly
- **Challenge**: Defining similarity metrics
- **Application**: Personalized recommendations, credit scoring

### Group Fairness
Statistical parity across different demographic groups
- **Demographic Parity**: Equal positive rates across groups
- **Equalized Odds**: Equal true positive and false positive rates
- **Equal Opportunity**: Equal true positive rates

### Procedural Fairness
Focus on the decision-making process rather than outcomes
- **Transparency**: Clear decision criteria and processes
- **Consistency**: Same process applied to all cases
- **Due Process**: Opportunity for appeal and explanation

### Distributive Fairness
Fair allocation of resources and outcomes
- **Need-based**: Allocation based on requirements
- **Merit-based**: Allocation based on qualifications
- **Equality**: Equal distribution regardless of characteristics

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import cvxpy as cp  # For optimization-based fairness

class FairnessImplementation:
    """Implementation of various fairness concepts and metrics"""
    
    def __init__(self):
        self.fairness_metrics = {}
        
    def generate_fair_dataset(self, n_samples=1000):
        """Generate dataset for fairness demonstration"""
        np.random.seed(42)
        
        # Protected attribute
        gender = np.random.binomial(1, 0.5, n_samples)
        
        # Qualifications (should be predictive)
        qualifications = np.random.normal(50, 20, n_samples)
        qualifications = np.clip(qualifications, 0, 100)
        
        # Add some correlation with gender (representing societal differences)
        qualifications += gender * 5  # Small advantage for one group
        
        # Target outcome based primarily on qualifications
        outcome_prob = 1 / (1 + np.exp(-(qualifications - 50) / 10))
        outcome = np.random.binomial(1, outcome_prob, n_samples)
        
        return pd.DataFrame({
            'gender': gender,
            'qualifications': qualifications,
            'outcome': outcome
        })
    
    def calculate_fairness_metrics(self, y_true, y_pred, protected_attr):
        """Calculate comprehensive fairness metrics"""
        
        # Overall metrics
        overall_accuracy = accuracy_score(y_true, y_pred)
        
        # Group-specific metrics
        groups = sorted(protected_attr.unique())
        group_metrics = {}
        
        for group in groups:
            mask = protected_attr == group
            group_true = y_true[mask]
            group_pred = y_pred[mask]
            
            if len(group_true) > 0:
                group_metrics[group] = {
                    'size': len(group_true),
                    'accuracy': accuracy_score(group_true, group_pred),
                    'precision': precision_score(group_true, group_pred, zero_division=0),
                    'recall': recall_score(group_true, group_pred, zero_division=0),
                    'positive_rate': np.mean(group_pred),
                    'base_rate': np.mean(group_true),
                    'tpr': recall_score(group_true, group_pred, zero_division=0),
                    'fpr': np.mean(group_pred[group_true == 0]) if np.sum(group_true == 0) > 0 else 0
                }
        
        # Fairness metrics
        fairness_metrics = {}
        
        if len(groups) >= 2:
            # Demographic Parity
            pos_rates = [group_metrics[g]['positive_rate'] for g in groups]
            fairness_metrics['demographic_parity_diff'] = max(pos_rates) - min(pos_rates)
            
            # Equal Opportunity
            tprs = [group_metrics[g]['tpr'] for g in groups]
            fairness_metrics['equal_opportunity_diff'] = max(tprs) - min(tprs)
            
            # Equalized Odds
            fprs = [group_metrics[g]['fpr'] for g in groups]
            fairness_metrics['equalized_odds_diff'] = max(tprs) - min(tprs) + max(fprs) - min(fprs)
            
            # Individual Fairness (simplified: variance in outcomes for similar qualifications)
            # This is a simplified proxy - true individual fairness requires similarity metrics
            fairness_metrics['individual_fairness_proxy'] = np.std(y_pred)
        
        return {
            'overall_accuracy': overall_accuracy,
            'group_metrics': group_metrics,
            'fairness_metrics': fairness_metrics
        }
    
    def implement_demographic_parity(self, X, y, protected_attr, model_class=RandomForestClassifier):
        """Implement demographic parity constraint using post-processing"""
        
        # Train base model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Get protected attributes for test set
        protected_test = protected_attr.iloc[X_test.index]
        
        # Train model
        model = model_class(random_state=42)
        model.fit(X_train, y_train)
        
        # Get predictions and probabilities
        y_pred_original = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Post-processing for demographic parity
        y_pred_fair = self.post_process_demographic_parity(
            y_prob, protected_test, target_rate=0.5
        )
        
        # Calculate metrics
        original_metrics = self.calculate_fairness_metrics(y_test, y_pred_original, protected_test)
        fair_metrics = self.calculate_fairness_metrics(y_test, y_pred_fair, protected_test)
        
        return {
            'model': model,
            'original_predictions': y_pred_original,
            'fair_predictions': y_pred_fair,
            'original_metrics': original_metrics,
            'fair_metrics': fair_metrics,
            'test_data': (X_test, y_test, protected_test)
        }
    
    def post_process_demographic_parity(self, y_prob, protected_attr, target_rate=0.5):
        """Post-process predictions to achieve demographic parity"""
        y_pred = np.zeros_like(y_prob, dtype=int)
        
        for group in protected_attr.unique():
            mask = protected_attr == group
            group_prob = y_prob[mask]
            
            # Sort by probability and select top target_rate proportion
            n_positive = int(len(group_prob) * target_rate)
            threshold_idx = np.argsort(group_prob)[-n_positive:]
            
            group_pred = np.zeros(len(group_prob), dtype=int)
            group_pred[threshold_idx] = 1
            
            y_pred[mask] = group_pred
        
        return y_pred
    
    def implement_equalized_odds(self, X, y, protected_attr):
        """Implement equalized odds using threshold optimization"""
        
        # Train base model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        protected_test = protected_attr.iloc[X_test.index]
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Get probabilities
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Find optimal thresholds for each group
        thresholds = self.optimize_equalized_odds_thresholds(
            y_test, y_prob, protected_test
        )
        
        # Apply group-specific thresholds
        y_pred_fair = np.zeros_like(y_prob, dtype=int)
        for group, threshold in thresholds.items():
            mask = protected_test == group
            y_pred_fair[mask] = (y_prob[mask] >= threshold).astype(int)
        
        # Compare with standard threshold
        y_pred_original = (y_prob >= 0.5).astype(int)
        
        original_metrics = self.calculate_fairness_metrics(y_test, y_pred_original, protected_test)
        fair_metrics = self.calculate_fairness_metrics(y_test, y_pred_fair, protected_test)
        
        return {
            'thresholds': thresholds,
            'original_metrics': original_metrics,
            'fair_metrics': fair_metrics
        }
    
    def optimize_equalized_odds_thresholds(self, y_true, y_prob, protected_attr):
        """Optimize thresholds to achieve equalized odds"""
        
        # Simple grid search for optimal thresholds
        groups = sorted(protected_attr.unique())
        best_thresholds = {}
        best_unfairness = float('inf')
        
        threshold_range = np.linspace(0.1, 0.9, 20)
        
        for t1 in threshold_range:
            for t2 in threshold_range:
                thresholds = {groups[0]: t1, groups[1]: t2}
                
                # Calculate TPR and FPR for each group
                tprs = []
                fprs = []
                
                for group in groups:
                    mask = protected_attr == group
                    group_true = y_true[mask]
                    group_pred = (y_prob[mask] >= thresholds[group]).astype(int)
                    
                    if len(group_true) > 0:
                        tpr = recall_score(group_true, group_pred, zero_division=0)
                        fpr = np.mean(group_pred[group_true == 0]) if np.sum(group_true == 0) > 0 else 0
                        tprs.append(tpr)
                        fprs.append(fpr)
                
                # Calculate unfairness (difference in TPR and FPR)
                if len(tprs) >= 2 and len(fprs) >= 2:
                    unfairness = abs(tprs[0] - tprs[1]) + abs(fprs[0] - fprs[1])
                    
                    if unfairness < best_unfairness:
                        best_unfairness = unfairness
                        best_thresholds = thresholds.copy()
        
        return best_thresholds
    
    def visualize_fairness_comparison(self, results):
        """Visualize fairness metrics comparison"""
        
        original = results['original_metrics']
        fair = results['fair_metrics']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        accuracies = {
            'Overall': [original['overall_accuracy'], fair['overall_accuracy']],
            'Group 0': [original['group_metrics'][0]['accuracy'], 
                       fair['group_metrics'][0]['accuracy']],
            'Group 1': [original['group_metrics'][1]['accuracy'], 
                       fair['group_metrics'][1]['accuracy']]
        }
        
        x = np.arange(len(accuracies))
        width = 0.35
        
        for i, (label, values) in enumerate(accuracies.items()):
            axes[0, 0].bar([i - width/2, i + width/2], values, width/2, 
                          label=['Original', 'Fair'] if i == 0 else "",
                          alpha=0.7)
        
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(accuracies.keys())
        if len(accuracies) > 0:
            axes[0, 0].legend(['Original', 'Fair'])
        
        # Positive rates
        pos_rates_orig = [original['group_metrics'][g]['positive_rate'] for g in [0, 1]]
        pos_rates_fair = [fair['group_metrics'][g]['positive_rate'] for g in [0, 1]]
        
        x = ['Group 0', 'Group 1']
        axes[0, 1].bar([i - 0.2 for i in range(len(x))], pos_rates_orig, 0.4, 
                      label='Original', alpha=0.7)
        axes[0, 1].bar([i + 0.2 for i in range(len(x))], pos_rates_fair, 0.4, 
                      label='Fair', alpha=0.7)
        axes[0, 1].set_title('Positive Rates by Group')
        axes[0, 1].set_ylabel('Positive Rate')
        axes[0, 1].set_xticks(range(len(x)))
        axes[0, 1].set_xticklabels(x)
        axes[0, 1].legend()
        
        # Fairness metrics
        fairness_names = ['Demographic Parity', 'Equal Opportunity', 'Equalized Odds']
        orig_fairness = [
            original['fairness_metrics'].get('demographic_parity_diff', 0),
            original['fairness_metrics'].get('equal_opportunity_diff', 0),
            original['fairness_metrics'].get('equalized_odds_diff', 0)
        ]
        fair_fairness = [
            fair['fairness_metrics'].get('demographic_parity_diff', 0),
            fair['fairness_metrics'].get('equal_opportunity_diff', 0),
            fair['fairness_metrics'].get('equalized_odds_diff', 0)
        ]
        
        x = np.arange(len(fairness_names))
        axes[1, 0].bar(x - 0.2, orig_fairness, 0.4, label='Original', alpha=0.7)
        axes[1, 0].bar(x + 0.2, fair_fairness, 0.4, label='Fair', alpha=0.7)
        axes[1, 0].set_title('Fairness Violations (Lower is Better)')
        axes[1, 0].set_ylabel('Difference')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(fairness_names, rotation=45)
        axes[1, 0].legend()
        
        # Trade-off visualization
        accuracy_loss = original['overall_accuracy'] - fair['overall_accuracy']
        fairness_gain = (original['fairness_metrics'].get('demographic_parity_diff', 0) - 
                        fair['fairness_metrics'].get('demographic_parity_diff', 0))
        
        axes[1, 1].scatter([accuracy_loss], [fairness_gain], s=100, alpha=0.7)
        axes[1, 1].set_xlabel('Accuracy Loss')
        axes[1, 1].set_ylabel('Fairness Gain')
        axes[1, 1].set_title('Accuracy-Fairness Trade-off')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Demonstrate fairness implementation
fairness_impl = FairnessImplementation()

# Generate dataset
print("=== FAIRNESS IN AI APPLICATIONS ===")
data = fairness_impl.generate_fair_dataset(1000)
print("Dataset generated with gender and qualifications...")

# Prepare features
X = data[['qualifications']]
y = data['outcome']
protected_attr = data['gender']

# Implement demographic parity
print("\n1. Demographic Parity Implementation:")
dp_results = fairness_impl.implement_demographic_parity(X, y, protected_attr)

print(f"Original Demographic Parity Difference: {dp_results['original_metrics']['fairness_metrics']['demographic_parity_diff']:.3f}")
print(f"Fair Demographic Parity Difference: {dp_results['fair_metrics']['fairness_metrics']['demographic_parity_diff']:.3f}")

# Implement equalized odds
print("\n2. Equalized Odds Implementation:")
eo_results = fairness_impl.implement_equalized_odds(X, y, protected_attr)

print(f"Original Equal Opportunity Difference: {eo_results['original_metrics']['fairness_metrics']['equal_opportunity_diff']:.3f}")
print(f"Fair Equal Opportunity Difference: {eo_results['fair_metrics']['fairness_metrics']['equal_opportunity_diff']:.3f}")

# Visualize results
fairness_impl.visualize_fairness_comparison(dp_results)
```

## Summary

This chapter covered fairness concepts and implementation in AI applications:

### Key Fairness Concepts:
1. **Individual Fairness**: Similar treatment for similar individuals
2. **Group Fairness**: Statistical parity across demographic groups
3. **Procedural Fairness**: Fair decision-making processes
4. **Distributive Fairness**: Fair allocation of resources and outcomes

### Fairness Metrics:
- **Demographic Parity**: Equal positive rates across groups
- **Equal Opportunity**: Equal true positive rates across groups
- **Equalized Odds**: Equal true positive and false positive rates
- **Individual Fairness**: Lipschitz continuity of decisions

### Implementation Approaches:
- **Pre-processing**: Modify training data to reduce bias
- **In-processing**: Add fairness constraints during training
- **Post-processing**: Adjust model outputs for fairness

### Best Practices:
- Choose appropriate fairness definitions for the application context
- Consider multiple fairness metrics simultaneously
- Understand and communicate accuracy-fairness trade-offs
- Engage stakeholders in defining fairness requirements
- Continuously monitor fairness in deployed systems
- Document fairness considerations and decisions

---

*Fairness in AI requires careful consideration of context, stakeholder needs, and the inherent trade-offs between different notions of fairness and model performance.* 