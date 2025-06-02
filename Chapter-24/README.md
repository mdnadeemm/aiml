# Chapter 24: AI Ethics and Responsibilities

## Learning Objectives
By the end of this chapter, students will be able to:
- Understand fundamental ethical principles in AI development
- Identify potential ethical issues in AI applications
- Apply ethical frameworks to AI decision-making
- Implement responsible AI practices in development
- Evaluate AI systems for ethical compliance and social impact

## Table of Contents
1. [Introduction to AI Ethics](#introduction)
2. [Ethical Frameworks](#frameworks)
3. [Bias and Fairness](#bias-fairness)
4. [Privacy and Surveillance](#privacy)
5. [Autonomy and Human Agency](#autonomy)
6. [Accountability and Transparency](#accountability)

## 1. Introduction to AI Ethics {#introduction}

As AI systems become increasingly powerful and ubiquitous, ethical considerations become paramount. AI ethics encompasses the moral principles and values that guide the development, deployment, and use of artificial intelligence systems.

### Key Ethical Concerns

**Bias and Discrimination**: AI systems may perpetuate or amplify existing societal biases
**Privacy**: Protection of personal data and surveillance concerns
**Transparency**: The "black box" problem and explainability
**Accountability**: Who is responsible when AI systems cause harm
**Human Agency**: Maintaining meaningful human control and choice
**Justice and Fairness**: Ensuring equitable outcomes across different groups

### Stakeholders in AI Ethics

**Developers and Engineers**: Technical implementation of ethical principles
**Companies and Organizations**: Corporate responsibility and governance
**Governments and Regulators**: Policy frameworks and legal compliance
**Users and Society**: Impact on individuals and communities
**Academic Researchers**: Studying ethical implications and solutions

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class EthicalAIAnalyzer:
    """Tool for analyzing ethical implications of AI systems"""
    
    def __init__(self):
        self.metrics = {}
        self.bias_metrics = {}
        
    def simulate_hiring_dataset(self, n_samples=1000):
        """Create synthetic hiring dataset to demonstrate bias"""
        np.random.seed(42)
        
        # Generate features
        age = np.random.normal(35, 10, n_samples)
        age = np.clip(age, 22, 65)
        
        # Gender (0: female, 1: male) - introduce bias
        gender = np.random.binomial(1, 0.4, n_samples)
        
        # Education score (0-100)
        education = np.random.normal(75, 15, n_samples)
        education = np.clip(education, 0, 100)
        
        # Experience years
        experience = np.random.poisson(8, n_samples)
        experience = np.clip(experience, 0, 30)
        
        # Interview score (biased towards males)
        interview_base = 0.3 * education + 0.2 * experience + np.random.normal(0, 10, n_samples)
        gender_bias = gender * 5  # Males get +5 bonus
        interview_score = interview_base + gender_bias
        interview_score = np.clip(interview_score, 0, 100)
        
        # Hiring decision (biased)
        hiring_prob = (0.01 * interview_score + 0.005 * education + 
                      0.02 * experience + 0.1 * gender - 0.5)
        hiring_prob = 1 / (1 + np.exp(-hiring_prob))  # Sigmoid
        hired = np.random.binomial(1, hiring_prob, n_samples)
        
        # Create DataFrame
        data = pd.DataFrame({
            'age': age,
            'gender': gender,
            'education_score': education,
            'experience_years': experience,
            'interview_score': interview_score,
            'hired': hired
        })
        
        return data
    
    def analyze_bias(self, data, protected_attribute='gender', target='hired'):
        """Analyze bias in dataset and model predictions"""
        
        # Group-based metrics
        groups = data[protected_attribute].unique()
        bias_analysis = {}
        
        for group in groups:
            group_data = data[data[protected_attribute] == group]
            group_name = f"Group_{group}"
            
            bias_analysis[group_name] = {
                'size': len(group_data),
                'positive_rate': group_data[target].mean(),
                'avg_interview_score': group_data['interview_score'].mean(),
                'avg_education': group_data['education_score'].mean(),
                'avg_experience': group_data['experience_years'].mean()
            }
        
        # Calculate disparate impact
        if len(groups) == 2:
            group_0_rate = bias_analysis[f"Group_{groups[0]}"]['positive_rate']
            group_1_rate = bias_analysis[f"Group_{groups[1]}"]['positive_rate']
            
            disparate_impact = min(group_0_rate, group_1_rate) / max(group_0_rate, group_1_rate)
            bias_analysis['disparate_impact'] = disparate_impact
            bias_analysis['disparate_impact_pass'] = disparate_impact >= 0.8  # 80% rule
        
        return bias_analysis
    
    def visualize_bias_analysis(self, data, bias_analysis):
        """Visualize bias analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Hiring rates by gender
        hiring_by_gender = data.groupby('gender')['hired'].agg(['count', 'sum', 'mean']).reset_index()
        hiring_by_gender['gender_label'] = hiring_by_gender['gender'].map({0: 'Female', 1: 'Male'})
        
        bars = axes[0, 0].bar(hiring_by_gender['gender_label'], hiring_by_gender['mean'], 
                             color=['pink', 'lightblue'], alpha=0.7)
        axes[0, 0].set_title('Hiring Rate by Gender')
        axes[0, 0].set_ylabel('Hiring Rate')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom')
        
        # Interview scores by gender
        axes[0, 1].boxplot([data[data['gender'] == 0]['interview_score'],
                           data[data['gender'] == 1]['interview_score']], 
                          labels=['Female', 'Male'])
        axes[0, 1].set_title('Interview Scores by Gender')
        axes[0, 1].set_ylabel('Interview Score')
        
        # Education vs hiring outcome
        hired_data = data[data['hired'] == 1]
        not_hired_data = data[data['hired'] == 0]
        
        axes[1, 0].scatter(not_hired_data['education_score'], not_hired_data['experience_years'], 
                          alpha=0.5, label='Not Hired', color='red')
        axes[1, 0].scatter(hired_data['education_score'], hired_data['experience_years'], 
                          alpha=0.5, label='Hired', color='green')
        axes[1, 0].set_xlabel('Education Score')
        axes[1, 0].set_ylabel('Experience Years')
        axes[1, 0].set_title('Education vs Experience (Hiring Outcomes)')
        axes[1, 0].legend()
        
        # Bias metrics summary
        metrics_text = f"Disparate Impact: {bias_analysis.get('disparate_impact', 'N/A'):.3f}\n"
        metrics_text += f"80% Rule Pass: {bias_analysis.get('disparate_impact_pass', 'N/A')}\n\n"
        
        for group, metrics in bias_analysis.items():
            if group.startswith('Group_'):
                metrics_text += f"{group}:\n"
                metrics_text += f"  Size: {metrics['size']}\n"
                metrics_text += f"  Hire Rate: {metrics['positive_rate']:.3f}\n"
                metrics_text += f"  Avg Interview: {metrics['avg_interview_score']:.1f}\n\n"
        
        axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                       verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Bias Analysis Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def implement_fairness_constraint(self, data, target_disparate_impact=0.9):
        """Demonstrate fairness constraint implementation"""
        print("=== IMPLEMENTING FAIRNESS CONSTRAINTS ===")
        
        # Train biased model
        features = ['age', 'education_score', 'experience_years', 'interview_score']
        X = data[features]
        y = data['hired']
        gender = data['gender']
        
        X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(
            X, y, gender, test_size=0.3, random_state=42, stratify=y
        )
        
        # Original biased model
        biased_model = RandomForestClassifier(random_state=42)
        biased_model.fit(X_train, y_train)
        biased_pred = biased_model.predict(X_test)
        
        # Calculate bias in original model
        original_bias = self.calculate_model_bias(y_test, biased_pred, gender_test)
        
        # Fairness-constrained approach (simplified post-processing)
        fair_pred = self.apply_fairness_postprocessing(
            biased_pred, gender_test, target_disparate_impact
        )
        
        # Calculate bias in fair model
        fair_bias = self.calculate_model_bias(y_test, fair_pred, gender_test)
        
        # Visualize comparison
        self.visualize_fairness_comparison(original_bias, fair_bias)
        
        return biased_model, original_bias, fair_bias
    
    def calculate_model_bias(self, y_true, y_pred, protected_attr):
        """Calculate bias metrics for model predictions"""
        results = {}
        
        for group in [0, 1]:
            mask = protected_attr == group
            group_pred = y_pred[mask]
            group_true = y_true[mask]
            
            results[f'group_{group}'] = {
                'predicted_positive_rate': np.mean(group_pred),
                'actual_positive_rate': np.mean(group_true),
                'accuracy': np.mean(group_pred == group_true),
                'size': len(group_pred)
            }
        
        # Disparate impact
        rate_0 = results['group_0']['predicted_positive_rate']
        rate_1 = results['group_1']['predicted_positive_rate']
        
        if rate_0 > 0 and rate_1 > 0:
            disparate_impact = min(rate_0, rate_1) / max(rate_0, rate_1)
        else:
            disparate_impact = 0
        
        results['disparate_impact'] = disparate_impact
        
        return results
    
    def apply_fairness_postprocessing(self, predictions, protected_attr, target_di=0.9):
        """Apply post-processing to improve fairness"""
        # Simple threshold adjustment approach
        fair_pred = predictions.copy()
        
        # Calculate current rates
        rate_0 = np.mean(predictions[protected_attr == 0])
        rate_1 = np.mean(predictions[protected_attr == 1])
        
        if rate_0 == 0 or rate_1 == 0:
            return fair_pred
        
        current_di = min(rate_0, rate_1) / max(rate_0, rate_1)
        
        if current_di < target_di:
            # Adjust the group with lower rate
            if rate_0 < rate_1:
                # Increase rate for group 0
                group_0_mask = protected_attr == 0
                n_to_flip = int((target_di * rate_1 - rate_0) * np.sum(group_0_mask))
                
                # Find group 0 instances that are currently 0 and flip some to 1
                candidates = np.where((group_0_mask) & (predictions == 0))[0]
                if len(candidates) >= n_to_flip:
                    flip_indices = np.random.choice(candidates, n_to_flip, replace=False)
                    fair_pred[flip_indices] = 1
            else:
                # Increase rate for group 1
                group_1_mask = protected_attr == 1
                n_to_flip = int((target_di * rate_0 - rate_1) * np.sum(group_1_mask))
                
                candidates = np.where((group_1_mask) & (predictions == 0))[0]
                if len(candidates) >= n_to_flip:
                    flip_indices = np.random.choice(candidates, n_to_flip, replace=False)
                    fair_pred[flip_indices] = 1
        
        return fair_pred
    
    def visualize_fairness_comparison(self, original_bias, fair_bias):
        """Compare bias metrics before and after fairness intervention"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Predicted positive rates
        groups = ['Female', 'Male']
        original_rates = [original_bias['group_0']['predicted_positive_rate'],
                         original_bias['group_1']['predicted_positive_rate']]
        fair_rates = [fair_bias['group_0']['predicted_positive_rate'],
                     fair_bias['group_1']['predicted_positive_rate']]
        
        x = np.arange(len(groups))
        width = 0.35
        
        bars1 = axes[0].bar(x - width/2, original_rates, width, label='Original Model', alpha=0.7)
        bars2 = axes[0].bar(x + width/2, fair_rates, width, label='Fair Model', alpha=0.7)
        
        axes[0].set_xlabel('Gender')
        axes[0].set_ylabel('Predicted Positive Rate')
        axes[0].set_title('Predicted Hiring Rates by Gender')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(groups)
        axes[0].legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom')
        
        # Disparate impact comparison
        di_metrics = ['Original Model', 'Fair Model']
        di_values = [original_bias['disparate_impact'], fair_bias['disparate_impact']]
        
        bars = axes[1].bar(di_metrics, di_values, color=['red', 'green'], alpha=0.7)
        axes[1].axhline(y=0.8, color='orange', linestyle='--', label='80% Rule Threshold')
        axes[1].set_ylabel('Disparate Impact')
        axes[1].set_title('Disparate Impact Comparison')
        axes[1].legend()
        
        # Add value labels
        for bar, value in zip(bars, di_values):
            axes[1].text(bar.get_x() + bar.get_width()/2., value + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Original Disparate Impact: {original_bias['disparate_impact']:.3f}")
        print(f"Fair Model Disparate Impact: {fair_bias['disparate_impact']:.3f}")
        print(f"Improvement: {fair_bias['disparate_impact'] - original_bias['disparate_impact']:.3f}")

# Demonstrate ethical AI analysis
analyzer = EthicalAIAnalyzer()

# Generate and analyze biased dataset
print("=== ETHICAL AI ANALYSIS DEMONSTRATION ===")
hiring_data = analyzer.simulate_hiring_dataset(1000)
print("Dataset created with inherent bias...")

# Analyze bias
bias_results = analyzer.analyze_bias(hiring_data)
print("\nBias Analysis Results:")
for key, value in bias_results.items():
    print(f"{key}: {value}")

# Visualize bias
analyzer.visualize_bias_analysis(hiring_data, bias_results)

# Implement fairness constraints
biased_model, original_bias, fair_bias = analyzer.implement_fairness_constraint(hiring_data)
```

## Summary

This chapter introduced fundamental concepts in AI ethics and responsibility:

### Key Ethical Principles:
1. **Fairness and Non-discrimination**: Ensuring equitable treatment across groups
2. **Transparency and Explainability**: Making AI decisions understandable
3. **Privacy and Data Protection**: Safeguarding personal information
4. **Human Agency**: Maintaining meaningful human control
5. **Accountability**: Establishing clear responsibility for AI outcomes

### Implementation Strategies:
- **Bias Detection**: Regular auditing of AI systems for discriminatory outcomes
- **Fairness Constraints**: Technical methods to improve algorithmic fairness
- **Ethical Review Processes**: Systematic evaluation of AI projects
- **Stakeholder Engagement**: Including diverse perspectives in AI development
- **Continuous Monitoring**: Ongoing assessment of AI system impacts

### Best Practices:
- Integrate ethical considerations from the beginning of AI projects
- Use diverse and representative datasets for training
- Implement transparency and explainability features
- Establish clear governance and accountability frameworks
- Regularly audit AI systems for bias and fairness
- Engage with affected communities and stakeholders

---

## Exercises

1. **Bias Audit**: Conduct bias analysis on a real-world dataset
2. **Fairness Metrics**: Implement different fairness metrics and compare results
3. **Ethical Framework**: Develop ethical guidelines for a specific AI application
4. **Case Study Analysis**: Analyze ethical implications of a published AI system
5. **Stakeholder Mapping**: Identify and analyze stakeholders for an AI project

---

*Ethical AI development requires ongoing commitment to fairness, transparency, and social responsibility throughout the entire AI lifecycle.* 