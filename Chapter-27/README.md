# Chapter 27: Mitigating Bias in Machine Learning

## Learning Objectives
By the end of this chapter, students will be able to:
- Implement bias mitigation techniques at different stages of ML pipeline
- Apply preprocessing, in-processing, and post-processing methods
- Design bias-aware data collection and model evaluation strategies
- Evaluate the effectiveness of bias mitigation approaches
- Build inclusive and representative ML systems

## Table of Contents
1. [Bias Mitigation Framework](#framework)
2. [Pre-processing Techniques](#preprocessing)
3. [In-processing Methods](#in-processing)
4. [Post-processing Approaches](#post-processing)
5. [Evaluation and Monitoring](#evaluation)
6. [Best Practices](#best-practices)

## 1. Bias Mitigation Framework {#framework}

Effective bias mitigation requires a systematic approach throughout the entire machine learning lifecycle.

### Mitigation Stages
- **Data Collection**: Ensuring representative and inclusive datasets
- **Pre-processing**: Modifying data before training
- **In-processing**: Incorporating fairness during model training
- **Post-processing**: Adjusting outputs after training
- **Deployment**: Monitoring and updating live systems

### Key Strategies
- **Representation**: Ensuring all groups are adequately represented
- **Feature Engineering**: Removing or transforming biased features
- **Algorithmic Constraints**: Adding fairness objectives to loss functions
- **Threshold Optimization**: Adjusting decision boundaries per group
- **Continuous Monitoring**: Tracking bias metrics over time

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

class BiasMitigationToolkit:
    """Comprehensive toolkit for bias mitigation"""
    
    def __init__(self):
        self.mitigation_results = {}
        
    def create_biased_dataset(self, n_samples=2000):
        """Create a dataset with multiple sources of bias"""
        np.random.seed(42)
        
        # Protected attributes
        gender = np.random.binomial(1, 0.3, n_samples)  # 30% female
        race = np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1])  # Majority, minority1, minority2
        
        # Legitimate features
        education = np.random.normal(12, 3, n_samples)
        education = np.clip(education, 8, 20)
        
        experience = np.random.exponential(5, n_samples)
        experience = np.clip(experience, 0, 30)
        
        skills_test = np.random.normal(75, 15, n_samples)
        skills_test = np.clip(skills_test, 0, 100)
        
        # Introduce bias
        # Historical bias in education
        education -= gender * 1.5 + (race == 1) * 1.0 + (race == 2) * 2.0
        education = np.clip(education, 8, 20)
        
        # Network effects in experience
        experience += (race == 0) * 2  # Majority group gets more opportunities
        experience = np.clip(experience, 0, 30)
        
        # Biased outcome
        hiring_score = (0.3 * education + 0.3 * experience + 0.4 * skills_test + 
                       gender * (-5) + (race == 0) * 10 + np.random.normal(0, 5, n_samples))
        
        hired = (hiring_score > np.percentile(hiring_score, 70)).astype(int)
        
        return pd.DataFrame({
            'gender': gender,
            'race': race,
            'education': education,
            'experience': experience,
            'skills_test': skills_test,
            'hired': hired
        })
    
    def calculate_bias_metrics(self, y_true, y_pred, protected_attr, attr_name="attr"):
        """Calculate comprehensive bias metrics"""
        groups = sorted(protected_attr.unique())
        metrics = {}
        
        # Group-specific metrics
        for group in groups:
            mask = protected_attr == group
            if np.sum(mask) > 0:
                group_true = y_true[mask]
                group_pred = y_pred[mask]
                
                metrics[f'{attr_name}_group_{group}'] = {
                    'size': len(group_true),
                    'positive_rate': np.mean(group_pred),
                    'accuracy': accuracy_score(group_true, group_pred),
                    'precision': precision_score(group_true, group_pred, zero_division=0),
                    'recall': recall_score(group_true, group_pred, zero_division=0)
                }
        
        # Fairness metrics
        if len(groups) >= 2:
            pos_rates = [metrics[f'{attr_name}_group_{g}']['positive_rate'] for g in groups if f'{attr_name}_group_{g}' in metrics]
            if len(pos_rates) >= 2:
                metrics[f'{attr_name}_demographic_parity'] = max(pos_rates) - min(pos_rates)
                if max(pos_rates) > 0:
                    metrics[f'{attr_name}_disparate_impact'] = min(pos_rates) / max(pos_rates)
                else:
                    metrics[f'{attr_name}_disparate_impact'] = 1.0
        
        return metrics
    
    def preprocess_data_balancing(self, data, target_col='hired', protected_cols=['gender', 'race']):
        """Apply various preprocessing techniques for bias mitigation"""
        
        results = {}
        
        # Original data
        X_orig = data.drop(columns=[target_col])
        y_orig = data[target_col]
        
        # 1. Representation balancing
        print("=== PREPROCESSING: REPRESENTATION BALANCING ===")
        
        # Balance by oversampling minority groups
        balanced_data = self.balance_representation(data, protected_cols[0])
        X_balanced = balanced_data.drop(columns=[target_col])
        y_balanced = balanced_data[target_col]
        
        results['original'] = (X_orig, y_orig)
        results['balanced'] = (X_balanced, y_balanced)
        
        # 2. Feature removal
        print("Removing potentially biased features...")
        X_fair = data.drop(columns=[target_col] + protected_cols)
        y_fair = data[target_col]
        results['feature_removed'] = (X_fair, y_fair)
        
        # 3. SMOTE for class balance
        print("Applying SMOTE for class balance...")
        features_for_smote = ['education', 'experience', 'skills_test']
        X_smote = data[features_for_smote]
        
        smote = SMOTE(random_state=42)
        X_smote_balanced, y_smote_balanced = smote.fit_resample(X_smote, y_orig)
        results['smote'] = (X_smote_balanced, y_smote_balanced)
        
        return results
    
    def balance_representation(self, data, protected_col):
        """Balance representation across protected groups"""
        groups = data[protected_col].unique()
        group_sizes = [len(data[data[protected_col] == g]) for g in groups]
        target_size = max(group_sizes)
        
        balanced_dfs = []
        for group in groups:
            group_data = data[data[protected_col] == group]
            if len(group_data) < target_size:
                # Oversample minority group
                oversampled = group_data.sample(n=target_size, replace=True, random_state=42)
                balanced_dfs.append(oversampled)
            else:
                balanced_dfs.append(group_data)
        
        return pd.concat(balanced_dfs, ignore_index=True)
    
    def train_with_fairness_constraints(self, X, y, protected_attr):
        """Implement in-processing fairness constraints"""
        
        print("=== IN-PROCESSING: FAIRNESS CONSTRAINTS ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        if hasattr(protected_attr, 'iloc'):
            protected_test = protected_attr.iloc[X_test.index]
        else:
            protected_test = protected_attr[X_test.index]
        
        # Standard model
        standard_model = RandomForestClassifier(random_state=42, n_estimators=100)
        standard_model.fit(X_train, y_train)
        y_pred_standard = standard_model.predict(X_test)
        
        # Fairness-aware model (simplified: class weights)
        # Calculate class weights to balance representation
        sample_weights = self.calculate_fairness_weights(y_train, protected_attr.iloc[X_train.index] if hasattr(protected_attr, 'iloc') else protected_attr[X_train.index])
        
        fair_model = RandomForestClassifier(random_state=42, n_estimators=100)
        fair_model.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred_fair = fair_model.predict(X_test)
        
        # Calculate bias metrics
        standard_metrics = self.calculate_bias_metrics(y_test, y_pred_standard, protected_test, "gender")
        fair_metrics = self.calculate_bias_metrics(y_test, y_pred_fair, protected_test, "gender")
        
        return {
            'standard_model': standard_model,
            'fair_model': fair_model,
            'standard_predictions': y_pred_standard,
            'fair_predictions': y_pred_fair,
            'standard_metrics': standard_metrics,
            'fair_metrics': fair_metrics,
            'test_data': (X_test, y_test, protected_test)
        }
    
    def calculate_fairness_weights(self, y, protected_attr):
        """Calculate sample weights to promote fairness"""
        weights = np.ones(len(y))
        
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            group_positive_rate = np.mean(y[group_mask])
            
            # Increase weight for underrepresented positive cases
            if group_positive_rate < 0.5:
                weights[group_mask & (y == 1)] *= 2.0
            
            # Decrease weight for overrepresented groups
            group_size = np.sum(group_mask)
            if group_size > len(y) * 0.6:  # If group is >60% of data
                weights[group_mask] *= 0.5
        
        return weights
    
    def postprocess_for_fairness(self, y_prob, protected_attr, method='threshold_optimization'):
        """Apply post-processing techniques for fairness"""
        
        print("=== POST-PROCESSING: FAIRNESS ADJUSTMENT ===")
        
        if method == 'threshold_optimization':
            return self.optimize_thresholds_for_fairness(y_prob, protected_attr)
        elif method == 'calibration':
            return self.calibrate_by_group(y_prob, protected_attr)
        
    def optimize_thresholds_for_fairness(self, y_prob, protected_attr, target_parity=0.05):
        """Optimize thresholds to achieve demographic parity"""
        
        groups = sorted(protected_attr.unique())
        thresholds = {}
        
        # Find thresholds that achieve similar positive rates
        overall_rate = np.mean(y_prob > 0.5)
        
        for group in groups:
            group_mask = protected_attr == group
            group_prob = y_prob[group_mask]
            
            # Find threshold that gives rate close to overall rate
            best_threshold = 0.5
            best_diff = float('inf')
            
            for threshold in np.linspace(0.1, 0.9, 50):
                rate = np.mean(group_prob > threshold)
                diff = abs(rate - overall_rate)
                
                if diff < best_diff:
                    best_diff = diff
                    best_threshold = threshold
            
            thresholds[group] = best_threshold
        
        # Apply group-specific thresholds
        y_pred_fair = np.zeros_like(y_prob, dtype=int)
        for group, threshold in thresholds.items():
            group_mask = protected_attr == group
            y_pred_fair[group_mask] = (y_prob[group_mask] > threshold).astype(int)
        
        return y_pred_fair, thresholds
    
    def evaluate_mitigation_effectiveness(self, data):
        """Comprehensive evaluation of bias mitigation techniques"""
        
        print("=== COMPREHENSIVE BIAS MITIGATION EVALUATION ===")
        
        # Prepare baseline
        X_baseline = data[['education', 'experience', 'skills_test']]
        y_baseline = data['hired']
        protected_attr = data['gender']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_baseline, y_baseline, test_size=0.3, random_state=42, stratify=y_baseline
        )
        protected_test = protected_attr.iloc[X_test.index]
        
        results = {}
        
        # 1. Baseline model
        baseline_model = RandomForestClassifier(random_state=42)
        baseline_model.fit(X_train, y_train)
        y_pred_baseline = baseline_model.predict(X_test)
        
        results['baseline'] = {
            'predictions': y_pred_baseline,
            'metrics': self.calculate_bias_metrics(y_test, y_pred_baseline, protected_test, "gender")
        }
        
        # 2. Preprocessing approach
        balanced_data = self.balance_representation(data, 'gender')
        X_balanced = balanced_data[['education', 'experience', 'skills_test']]
        y_balanced = balanced_data['hired']
        
        preprocessing_model = RandomForestClassifier(random_state=42)
        preprocessing_model.fit(X_balanced, y_balanced)
        y_pred_preprocessing = preprocessing_model.predict(X_test)
        
        results['preprocessing'] = {
            'predictions': y_pred_preprocessing,
            'metrics': self.calculate_bias_metrics(y_test, y_pred_preprocessing, protected_test, "gender")
        }
        
        # 3. In-processing approach
        sample_weights = self.calculate_fairness_weights(y_train, protected_attr.iloc[X_train.index])
        inprocessing_model = RandomForestClassifier(random_state=42)
        inprocessing_model.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred_inprocessing = inprocessing_model.predict(X_test)
        
        results['inprocessing'] = {
            'predictions': y_pred_inprocessing,
            'metrics': self.calculate_bias_metrics(y_test, y_pred_inprocessing, protected_test, "gender")
        }
        
        # 4. Post-processing approach
        y_prob_baseline = baseline_model.predict_proba(X_test)[:, 1]
        y_pred_postprocessing, thresholds = self.optimize_thresholds_for_fairness(y_prob_baseline, protected_test)
        
        results['postprocessing'] = {
            'predictions': y_pred_postprocessing,
            'metrics': self.calculate_bias_metrics(y_test, y_pred_postprocessing, protected_test, "gender"),
            'thresholds': thresholds
        }
        
        return results, (X_test, y_test, protected_test)
    
    def visualize_mitigation_comparison(self, results, test_data):
        """Visualize the effectiveness of different mitigation approaches"""
        
        X_test, y_test, protected_test = test_data
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        methods = list(results.keys())
        
        # 1. Accuracy comparison
        accuracies = []
        for method in methods:
            y_pred = results[method]['predictions']
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
        
        axes[0, 0].bar(methods, accuracies, alpha=0.7)
        axes[0, 0].set_title('Overall Accuracy by Method')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Fairness metrics
        fairness_metrics = []
        for method in methods:
            dp_diff = results[method]['metrics'].get('gender_demographic_parity', 0)
            fairness_metrics.append(dp_diff)
        
        bars = axes[0, 1].bar(methods, fairness_metrics, alpha=0.7, color='orange')
        axes[0, 1].set_title('Demographic Parity Difference (Lower is Better)')
        axes[0, 1].set_ylabel('Difference in Positive Rates')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, fairness_metrics):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., value + 0.001,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Group-specific positive rates
        group_0_rates = []
        group_1_rates = []
        
        for method in methods:
            metrics = results[method]['metrics']
            group_0_rates.append(metrics.get('gender_group_0', {}).get('positive_rate', 0))
            group_1_rates.append(metrics.get('gender_group_1', {}).get('positive_rate', 0))
        
        x = np.arange(len(methods))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, group_0_rates, width, label='Male', alpha=0.7)
        axes[1, 0].bar(x + width/2, group_1_rates, width, label='Female', alpha=0.7)
        axes[1, 0].set_title('Positive Rates by Gender')
        axes[1, 0].set_ylabel('Positive Rate')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(methods, rotation=45)
        axes[1, 0].legend()
        
        # 4. Accuracy-Fairness tradeoff
        baseline_accuracy = accuracies[0]
        baseline_fairness = fairness_metrics[0]
        
        accuracy_changes = [acc - baseline_accuracy for acc in accuracies]
        fairness_changes = [baseline_fairness - fair for fair in fairness_metrics]
        
        colors = ['red', 'blue', 'green', 'orange'][:len(methods)]
        
        for i, method in enumerate(methods):
            axes[1, 1].scatter(accuracy_changes[i], fairness_changes[i], 
                              s=100, alpha=0.7, c=colors[i], label=method)
        
        axes[1, 1].set_xlabel('Accuracy Change from Baseline')
        axes[1, 1].set_ylabel('Fairness Improvement')
        axes[1, 1].set_title('Accuracy-Fairness Tradeoff')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[1, 1].axvline(x=0, color='black', linestyle='--', alpha=0.3)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\n=== MITIGATION EFFECTIVENESS SUMMARY ===")
        for method in methods:
            acc = accuracy_score(y_test, results[method]['predictions'])
            dp_diff = results[method]['metrics'].get('gender_demographic_parity', 0)
            print(f"{method.upper()}:")
            print(f"  Accuracy: {acc:.3f}")
            print(f"  Demographic Parity Difference: {dp_diff:.3f}")
            print()

# Demonstrate comprehensive bias mitigation
toolkit = BiasMitigationToolkit()

# Create biased dataset
biased_data = toolkit.create_biased_dataset(2000)
print("Created biased dataset with gender and race bias...")

# Evaluate all mitigation approaches
mitigation_results, test_data = toolkit.evaluate_mitigation_effectiveness(biased_data)

# Visualize comparison
toolkit.visualize_mitigation_comparison(mitigation_results, test_data)
```

## Summary

This chapter provided a comprehensive approach to mitigating bias in machine learning:

### Mitigation Approaches:
1. **Pre-processing**: Data balancing, feature selection, representation improvement
2. **In-processing**: Fairness constraints, sample weighting, adversarial training
3. **Post-processing**: Threshold optimization, output calibration

### Key Techniques:
- **Data Augmentation**: Oversampling underrepresented groups
- **Feature Engineering**: Removing or transforming biased features
- **Algorithmic Fairness**: Incorporating fairness objectives
- **Threshold Tuning**: Group-specific decision boundaries
- **Ensemble Methods**: Combining multiple fairness approaches

### Best Practices:
- Apply multiple mitigation techniques for robust results
- Monitor bias metrics throughout the ML lifecycle
- Balance fairness improvements with accuracy requirements
- Engage diverse stakeholders in bias assessment
- Document mitigation efforts and trade-offs
- Continuously update systems as new biases emerge

---

*Effective bias mitigation requires a systematic, multi-stage approach that considers the specific context and requirements of each application.* 