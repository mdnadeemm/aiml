# Chapter 25: Understanding Bias in AI Models

## Learning Objectives
By the end of this chapter, students will be able to:
- Identify different types of bias in AI systems and data
- Understand the sources and causes of algorithmic bias
- Implement bias detection techniques and metrics
- Apply bias mitigation strategies during model development
- Evaluate the effectiveness of bias reduction methods

## Table of Contents
1. [Types of Bias in AI](#types-of-bias)
2. [Sources of Bias](#sources)
3. [Bias Detection Methods](#detection)
4. [Bias Metrics and Evaluation](#metrics)
5. [Mitigation Strategies](#mitigation)
6. [Case Studies](#case-studies)

## 1. Types of Bias in AI {#types-of-bias}

Bias in AI systems can manifest in various forms, each with different implications for fairness and model performance.

### Historical Bias
Reflects past discrimination and inequalities present in training data
- **Example**: Historical hiring data showing gender bias in certain roles
- **Impact**: Models learn and perpetuate past discriminatory patterns

### Representation Bias
Occurs when certain groups are underrepresented in training data
- **Example**: Medical datasets primarily containing data from one demographic
- **Impact**: Poor performance for underrepresented groups

### Measurement Bias
Arises from differences in data quality or collection methods across groups
- **Example**: Credit scores measured differently across communities
- **Impact**: Systematic errors in model predictions for certain groups

### Evaluation Bias
Results from inappropriate evaluation metrics or benchmarks
- **Example**: Using accuracy alone when classes are imbalanced
- **Impact**: Models appear fair but perform poorly for minority classes

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

class BiasAnalyzer:
    """Comprehensive bias analysis toolkit"""
    
    def __init__(self):
        self.bias_metrics = {}
        self.models = {}
        
    def create_biased_dataset(self, n_samples=2000):
        """Create synthetic dataset with multiple types of bias"""
        np.random.seed(42)
        
        # Protected attributes
        gender = np.random.binomial(1, 0.3, n_samples)  # 30% female (representation bias)
        race = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])  # Majority, minority1, minority2
        age = np.random.normal(40, 12, n_samples)
        age = np.clip(age, 22, 70)
        
        # Features with bias
        # Education - biased by gender and race (historical bias)
        education_base = np.random.normal(12, 3, n_samples)
        education = education_base + gender * (-1.5) + (race == 1) * (-1) + (race == 2) * (-2)
        education = np.clip(education, 8, 20)
        
        # Income - measurement bias (different scales for different groups)
        income_base = 30000 + 2000 * education + 500 * age + np.random.normal(0, 10000, n_samples)
        # Measurement bias: income reported differently by groups
        measurement_noise = np.where(race == 0, 
                                   np.random.normal(0, 2000, n_samples),  # Accurate reporting
                                   np.random.normal(-5000, 5000, n_samples))  # Underreporting + noise
        income = income_base + measurement_noise
        income = np.clip(income, 20000, 200000)
        
        # Credit score - historical bias
        credit_base = 650 + 10 * education + income/1000 + np.random.normal(0, 50, n_samples)
        historical_bias = gender * (-30) + (race == 1) * (-40) + (race == 2) * (-60)
        credit_score = credit_base + historical_bias
        credit_score = np.clip(credit_score, 300, 850)
        
        # Target: loan approval (biased)
        approval_prob = (credit_score - 500) / 500
        approval_prob += gender * 0.1 + (race == 0) * 0.15  # Direct discrimination
        approval_prob = 1 / (1 + np.exp(-3 * (approval_prob - 0.5)))
        loan_approved = np.random.binomial(1, approval_prob, n_samples)
        
        return pd.DataFrame({
            'gender': gender,  # 0: male, 1: female
            'race': race,      # 0: majority, 1: minority1, 2: minority2
            'age': age,
            'education': education,
            'income': income,
            'credit_score': credit_score,
            'loan_approved': loan_approved
        })
    
    def visualize_bias_patterns(self, data):
        """Visualize different types of bias in the dataset"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Representation bias
        gender_counts = data['gender'].value_counts()
        race_counts = data['race'].value_counts()
        
        axes[0, 0].bar(['Male', 'Female'], [gender_counts[0], gender_counts[1]], 
                      color=['blue', 'pink'], alpha=0.7)
        axes[0, 0].set_title('Gender Representation')
        axes[0, 0].set_ylabel('Count')
        
        axes[0, 1].bar(['Majority', 'Minority 1', 'Minority 2'], 
                      [race_counts[0], race_counts[1], race_counts[2]], 
                      color=['green', 'orange', 'red'], alpha=0.7)
        axes[0, 1].set_title('Race Representation')
        axes[0, 1].set_ylabel('Count')
        
        # Historical bias in education
        education_by_demo = data.groupby(['gender', 'race'])['education'].mean().reset_index()
        demo_labels = [f"G{row['gender']}_R{row['race']}" for _, row in education_by_demo.iterrows()]
        
        axes[0, 2].bar(demo_labels, education_by_demo['education'], alpha=0.7)
        axes[0, 2].set_title('Average Education by Demographics')
        axes[0, 2].set_ylabel('Years of Education')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Measurement bias in income
        for race in [0, 1, 2]:
            race_data = data[data['race'] == race]['income']
            axes[1, 0].hist(race_data, alpha=0.5, bins=30, 
                           label=f'Race {race}', density=True)
        axes[1, 0].set_title('Income Distribution by Race (Measurement Bias)')
        axes[1, 0].set_xlabel('Income')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        
        # Credit score bias
        credit_by_gender = data.groupby('gender')['credit_score'].mean()
        axes[1, 1].bar(['Male', 'Female'], credit_by_gender.values, 
                      color=['blue', 'pink'], alpha=0.7)
        axes[1, 1].set_title('Average Credit Score by Gender')
        axes[1, 1].set_ylabel('Credit Score')
        
        # Outcome bias
        approval_rates = data.groupby(['gender', 'race'])['loan_approved'].mean().reset_index()
        approval_labels = [f"G{row['gender']}_R{row['race']}" for _, row in approval_rates.iterrows()]
        
        axes[1, 2].bar(approval_labels, approval_rates['loan_approved'], alpha=0.7)
        axes[1, 2].set_title('Loan Approval Rates by Demographics')
        axes[1, 2].set_ylabel('Approval Rate')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def calculate_bias_metrics(self, y_true, y_pred, protected_attr, attr_name="protected_attr"):
        """Calculate comprehensive bias metrics"""
        metrics = {}
        
        # Get unique groups
        groups = sorted(protected_attr.unique())
        
        # Per-group metrics
        for group in groups:
            mask = protected_attr == group
            group_true = y_true[mask]
            group_pred = y_pred[mask]
            
            metrics[f'{attr_name}_group_{group}'] = {
                'size': len(group_true),
                'accuracy': accuracy_score(group_true, group_pred),
                'precision': precision_score(group_true, group_pred, zero_division=0),
                'recall': recall_score(group_true, group_pred, zero_division=0),
                'positive_rate': np.mean(group_pred),
                'base_rate': np.mean(group_true)
            }
        
        # Fairness metrics
        if len(groups) >= 2:
            # Demographic parity (equal positive rates)
            pos_rates = [metrics[f'{attr_name}_group_{g}']['positive_rate'] for g in groups]
            metrics[f'{attr_name}_demographic_parity'] = max(pos_rates) - min(pos_rates)
            
            # Equal opportunity (equal recall/TPR)
            recalls = [metrics[f'{attr_name}_group_{g}']['recall'] for g in groups]
            metrics[f'{attr_name}_equal_opportunity'] = max(recalls) - min(recalls)
            
            # Equalized odds (equal TPR and FPR)
            # For simplicity, we'll use the recall difference as proxy
            metrics[f'{attr_name}_equalized_odds'] = max(recalls) - min(recalls)
            
            # Disparate impact (ratio of positive rates)
            min_rate = min(pos_rates)
            max_rate = max(pos_rates)
            if max_rate > 0:
                metrics[f'{attr_name}_disparate_impact'] = min_rate / max_rate
            else:
                metrics[f'{attr_name}_disparate_impact'] = 1.0
        
        return metrics
    
    def train_and_evaluate_models(self, data):
        """Train models and evaluate bias"""
        print("=== TRAINING AND EVALUATING MODELS FOR BIAS ===")
        
        # Prepare data
        features = ['age', 'education', 'income', 'credit_score']
        X = data[features]
        y = data['loan_approved']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Get protected attributes for test set
        test_indices = X_test.index
        gender_test = data.loc[test_indices, 'gender']
        race_test = data.loc[test_indices, 'race']
        
        # Train models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Scale features for logistic regression
            if name == 'Logistic Regression':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Overall performance
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Overall Accuracy: {accuracy:.3f}")
            
            # Bias analysis
            gender_metrics = self.calculate_bias_metrics(y_test, y_pred, gender_test, "gender")
            race_metrics = self.calculate_bias_metrics(y_test, y_pred, race_test, "race")
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'accuracy': accuracy,
                'gender_metrics': gender_metrics,
                'race_metrics': race_metrics
            }
            
            # Print bias metrics
            print(f"Gender Demographic Parity Difference: {gender_metrics['gender_demographic_parity']:.3f}")
            print(f"Gender Disparate Impact: {gender_metrics['gender_disparate_impact']:.3f}")
            print(f"Race Demographic Parity Difference: {race_metrics['race_demographic_parity']:.3f}")
            print(f"Race Disparate Impact: {race_metrics['race_disparate_impact']:.3f}")
        
        return results
    
    def visualize_model_bias(self, results, data):
        """Visualize bias metrics across models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        model_names = list(results.keys())
        
        # Gender bias metrics
        gender_dp = [results[m]['gender_metrics']['gender_demographic_parity'] for m in model_names]
        gender_di = [results[m]['gender_metrics']['gender_disparate_impact'] for m in model_names]
        
        axes[0, 0].bar(model_names, gender_dp, alpha=0.7, color='lightcoral')
        axes[0, 0].set_title('Gender Demographic Parity Difference')
        axes[0, 0].set_ylabel('Difference in Positive Rates')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(model_names, gender_di, alpha=0.7, color='lightblue')
        axes[0, 1].axhline(y=0.8, color='red', linestyle='--', label='80% Rule')
        axes[0, 1].set_title('Gender Disparate Impact')
        axes[0, 1].set_ylabel('Min Rate / Max Rate')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend()
        
        # Race bias metrics
        race_dp = [results[m]['race_metrics']['race_demographic_parity'] for m in model_names]
        race_di = [results[m]['race_metrics']['race_disparate_impact'] for m in model_names]
        
        axes[1, 0].bar(model_names, race_dp, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Race Demographic Parity Difference')
        axes[1, 0].set_ylabel('Difference in Positive Rates')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        axes[1, 1].bar(model_names, race_di, alpha=0.7, color='orange')
        axes[1, 1].axhline(y=0.8, color='red', linestyle='--', label='80% Rule')
        axes[1, 1].set_title('Race Disparate Impact')
        axes[1, 1].set_ylabel('Min Rate / Max Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def implement_bias_mitigation(self, data):
        """Demonstrate bias mitigation techniques"""
        print("\n=== BIAS MITIGATION TECHNIQUES ===")
        
        # Technique 1: Data preprocessing - Resampling
        print("1. Data Preprocessing: Balancing representation")
        
        # Balance gender representation
        male_data = data[data['gender'] == 0]
        female_data = data[data['gender'] == 1]
        
        # Oversample minority group
        min_size = min(len(male_data), len(female_data))
        max_size = max(len(male_data), len(female_data))
        
        if len(female_data) < len(male_data):
            # Oversample females
            female_oversampled = female_data.sample(n=max_size, replace=True, random_state=42)
            balanced_data = pd.concat([male_data, female_oversampled])
        else:
            # Oversample males
            male_oversampled = male_data.sample(n=max_size, replace=True, random_state=42)
            balanced_data = pd.concat([female_data, male_oversampled])
        
        print(f"Original gender distribution: {data['gender'].value_counts().to_dict()}")
        print(f"Balanced gender distribution: {balanced_data['gender'].value_counts().to_dict()}")
        
        # Technique 2: Feature modification
        print("\n2. Feature Engineering: Removing discriminatory features")
        
        # Train model without protected attributes
        features_fair = ['age', 'education', 'income', 'credit_score']
        X_fair = balanced_data[features_fair]
        y_fair = balanced_data['loan_approved']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_fair, y_fair, test_size=0.3, random_state=42, stratify=y_fair
        )
        
        # Get protected attributes for test set
        test_indices = X_test.index
        gender_test = balanced_data.loc[test_indices, 'gender']
        
        # Train fair model
        fair_model = RandomForestClassifier(random_state=42, n_estimators=100)
        fair_model.fit(X_train, y_train)
        y_pred_fair = fair_model.predict(X_test)
        
        # Evaluate fairness
        fair_metrics = self.calculate_bias_metrics(y_test, y_pred_fair, gender_test, "gender")
        
        print(f"Fair Model - Gender Disparate Impact: {fair_metrics['gender_disparate_impact']:.3f}")
        print(f"Fair Model - Demographic Parity Difference: {fair_metrics['gender_demographic_parity']:.3f}")
        
        return balanced_data, fair_model, fair_metrics

# Demonstrate bias analysis
analyzer = BiasAnalyzer()

# Create biased dataset
print("Creating biased dataset...")
biased_data = analyzer.create_biased_dataset(2000)

# Visualize bias patterns
analyzer.visualize_bias_patterns(biased_data)

# Train and evaluate models
model_results = analyzer.train_and_evaluate_models(biased_data)

# Visualize model bias
analyzer.visualize_model_bias(model_results, biased_data)

# Implement bias mitigation
balanced_data, fair_model, fair_metrics = analyzer.implement_bias_mitigation(biased_data)
```

## Summary

This chapter provided comprehensive coverage of bias in AI models:

### Types of Bias:
1. **Historical Bias**: Past discrimination reflected in data
2. **Representation Bias**: Unequal representation of groups
3. **Measurement Bias**: Systematic errors in data collection
4. **Evaluation Bias**: Inappropriate metrics or benchmarks
5. **Aggregation Bias**: Assuming one model fits all groups
6. **Confirmation Bias**: Seeking evidence that confirms preconceptions

### Detection Methods:
- Statistical parity testing
- Disparate impact analysis
- Equalized odds evaluation
- Individual fairness assessment
- Intersectional bias analysis

### Mitigation Strategies:
- **Pre-processing**: Data augmentation, resampling, feature selection
- **In-processing**: Fairness constraints, adversarial training
- **Post-processing**: Threshold adjustment, output modification
- **Hybrid approaches**: Combining multiple techniques

### Best Practices:
- Conduct bias audits throughout the ML lifecycle
- Use multiple fairness metrics for comprehensive evaluation
- Engage diverse stakeholders in bias assessment
- Document bias analysis and mitigation efforts
- Monitor deployed models for emerging bias
- Consider trade-offs between fairness and accuracy

---

## Exercises

1. **Bias Detection**: Analyze a real dataset for multiple types of bias
2. **Fairness Metrics**: Implement and compare different fairness metrics
3. **Mitigation Comparison**: Test various bias mitigation techniques
4. **Intersectional Analysis**: Study bias across multiple protected attributes
5. **Real-world Case Study**: Investigate bias in a deployed AI system

---

*Understanding and addressing bias in AI models is crucial for developing fair, equitable, and trustworthy AI systems that serve all members of society.* 