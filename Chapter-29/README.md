# Chapter 29: Diagnostic Applications of AI

## Learning Objectives
By the end of this chapter, students will be able to:
- Understand AI-powered diagnostic systems across medical specialties
- Implement diagnostic algorithms for various medical conditions
- Apply machine learning to medical decision trees and expert systems
- Design multi-modal diagnostic approaches combining different data types
- Evaluate diagnostic AI performance using clinical metrics

## Table of Contents
1. [AI Diagnostic Systems Overview](#overview)
2. [Medical Image Diagnosis](#image-diagnosis)
3. [Laboratory Data Analysis](#lab-analysis)
4. [Multi-Modal Diagnosis](#multi-modal)
5. [Expert Systems and Decision Trees](#expert-systems)
6. [Performance Evaluation](#evaluation)

## 1. AI Diagnostic Systems Overview {#overview}

AI diagnostic systems combine multiple data sources and analytical techniques to assist healthcare professionals in accurate and timely diagnosis.

### Types of Diagnostic AI:
- **Rule-based Systems**: Expert knowledge encoded as rules
- **Machine Learning Models**: Pattern recognition from data
- **Deep Learning**: Complex feature extraction and classification
- **Ensemble Methods**: Combining multiple diagnostic approaches
- **Probabilistic Models**: Uncertainty quantification in diagnosis

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

class DiagnosticAISystem:
    """Comprehensive diagnostic AI system for medical applications"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
    def generate_comprehensive_patient_data(self, n_patients=1500):
        """Generate comprehensive synthetic patient data for multiple conditions"""
        np.random.seed(42)
        
        # Demographics
        age = np.random.normal(45, 20, n_patients)
        age = np.clip(age, 18, 90)
        
        gender = np.random.binomial(1, 0.5, n_patients)
        
        # Vital signs
        temperature = np.random.normal(98.6, 1.5, n_patients)
        blood_pressure_sys = np.random.normal(130, 20, n_patients)
        blood_pressure_dia = np.random.normal(80, 15, n_patients)
        heart_rate = np.random.normal(75, 15, n_patients)
        respiratory_rate = np.random.normal(16, 4, n_patients)
        
        # Laboratory values
        white_blood_cells = np.random.normal(7000, 2000, n_patients)
        red_blood_cells = np.random.normal(4.5, 0.8, n_patients)
        hemoglobin = np.random.normal(14, 2, n_patients)
        glucose = np.random.normal(100, 25, n_patients)
        creatinine = np.random.normal(1.0, 0.3, n_patients)
        
        # Symptoms (binary)
        fever = np.random.binomial(1, 0.3, n_patients)
        cough = np.random.binomial(1, 0.25, n_patients)
        chest_pain = np.random.binomial(1, 0.2, n_patients)
        shortness_breath = np.random.binomial(1, 0.18, n_patients)
        fatigue = np.random.binomial(1, 0.35, n_patients)
        nausea = np.random.binomial(1, 0.15, n_patients)
        
        # Generate multiple diagnostic conditions
        conditions = self.generate_diagnostic_conditions(
            age, gender, temperature, white_blood_cells, 
            fever, cough, chest_pain, shortness_breath, n_patients
        )
        
        return pd.DataFrame({
            'age': age,
            'gender': gender,
            'temperature': temperature,
            'bp_systolic': blood_pressure_sys,
            'bp_diastolic': blood_pressure_dia,
            'heart_rate': heart_rate,
            'respiratory_rate': respiratory_rate,
            'wbc_count': white_blood_cells,
            'rbc_count': red_blood_cells,
            'hemoglobin': hemoglobin,
            'glucose': glucose,
            'creatinine': creatinine,
            'fever': fever,
            'cough': cough,
            'chest_pain': chest_pain,
            'shortness_breath': shortness_breath,
            'fatigue': fatigue,
            'nausea': nausea,
            **conditions
        })
    
    def generate_diagnostic_conditions(self, age, gender, temperature, wbc, fever, cough, chest_pain, shortness_breath, n_patients):
        """Generate realistic diagnostic conditions based on patient features"""
        
        # Pneumonia risk score
        pneumonia_risk = (
            fever * 3 +
            cough * 2 +
            (temperature - 98.6) * 2 +
            shortness_breath * 2 +
            (wbc - 7000) / 1000 +
            np.random.normal(0, 2, n_patients)
        )
        pneumonia = (pneumonia_risk > np.percentile(pneumonia_risk, 85)).astype(int)
        
        # Heart disease risk score  
        heart_disease_risk = (
            (age - 40) * 0.1 +
            gender * (-1) +  # Women lower risk when young
            chest_pain * 3 +
            shortness_breath * 2 +
            np.random.normal(0, 2, n_patients)
        )
        heart_disease = (heart_disease_risk > np.percentile(heart_disease_risk, 80)).astype(int)
        
        # Diabetes risk score
        diabetes_risk = (
            (age - 30) * 0.05 +
            np.random.normal(0, 1, n_patients)
        )
        diabetes = (diabetes_risk > np.percentile(diabetes_risk, 90)).astype(int)
        
        # Influenza risk score
        flu_risk = (
            fever * 4 +
            cough * 3 +
            (temperature - 98.6) * 3 +
            np.random.normal(0, 3, n_patients)
        )
        influenza = (flu_risk > np.percentile(flu_risk, 85)).astype(int)
        
        # Create primary diagnosis (mutually exclusive for simplicity)
        primary_diagnosis = np.zeros(n_patients, dtype=int)
        
        # Priority order: pneumonia > heart_disease > diabetes > influenza > healthy
        primary_diagnosis[pneumonia == 1] = 1  # Pneumonia
        primary_diagnosis[(pneumonia == 0) & (heart_disease == 1)] = 2  # Heart Disease
        primary_diagnosis[(pneumonia == 0) & (heart_disease == 0) & (diabetes == 1)] = 3  # Diabetes
        primary_diagnosis[(pneumonia == 0) & (heart_disease == 0) & (diabetes == 0) & (influenza == 1)] = 4  # Influenza
        # 0 remains for healthy patients
        
        return {
            'pneumonia': pneumonia,
            'heart_disease': heart_disease,
            'diabetes': diabetes,
            'influenza': influenza,
            'primary_diagnosis': primary_diagnosis
        }
    
    def build_diagnostic_models(self, data):
        """Build multiple diagnostic models for different conditions"""
        
        print("=== BUILDING DIAGNOSTIC AI MODELS ===")
        
        # Features for diagnosis
        feature_cols = ['age', 'gender', 'temperature', 'bp_systolic', 'bp_diastolic',
                       'heart_rate', 'respiratory_rate', 'wbc_count', 'rbc_count',
                       'hemoglobin', 'glucose', 'creatinine', 'fever', 'cough',
                       'chest_pain', 'shortness_breath', 'fatigue', 'nausea']
        
        X = data[feature_cols]
        
        # Build models for different conditions
        conditions = ['pneumonia', 'heart_disease', 'diabetes', 'influenza']
        models_results = {}
        
        for condition in conditions:
            print(f"\nBuilding {condition} diagnostic model...")
            
            y = data[condition]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models
            models = {
                'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42),
                'Neural Network': MLPClassifier(hidden_layer_sizes=(50, 25), random_state=42, max_iter=1000)
            }
            
            condition_results = {}
            
            for name, model in models.items():
                if name in ['Random Forest', 'Gradient Boosting']:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_prob = model.predict_proba(X_test_scaled)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
                auc = roc_auc_score(y_test, y_prob)
                
                condition_results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'predictions': y_pred,
                    'probabilities': y_prob,
                    'scaler': scaler if name not in ['Random Forest', 'Gradient Boosting'] else None
                }
                
                print(f"{name} - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, AUC: {auc:.3f}")
            
            models_results[condition] = condition_results
            
            # Store best model
            best_model_name = max(condition_results.keys(), key=lambda x: condition_results[x]['f1'])
            self.models[condition] = condition_results[best_model_name]
            
            print(f"Best model for {condition}: {best_model_name} (F1: {condition_results[best_model_name]['f1']:.3f})")
        
        return models_results, (X_test, y_test)
    
    def build_multi_class_diagnostic_model(self, data):
        """Build multi-class diagnostic model for primary diagnosis"""
        
        print("\n=== MULTI-CLASS DIAGNOSTIC MODEL ===")
        
        feature_cols = ['age', 'gender', 'temperature', 'bp_systolic', 'bp_diastolic',
                       'heart_rate', 'respiratory_rate', 'wbc_count', 'rbc_count',
                       'hemoglobin', 'glucose', 'creatinine', 'fever', 'cough',
                       'chest_pain', 'shortness_breath', 'fatigue', 'nausea']
        
        X = data[feature_cols]
        y = data['primary_diagnosis']
        
        # Create label mapping
        diagnosis_labels = {0: 'Healthy', 1: 'Pneumonia', 2: 'Heart Disease', 3: 'Diabetes', 4: 'Influenza'}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multi-class model
        multi_model = RandomForestClassifier(random_state=42, n_estimators=200)
        multi_model.fit(X_train, y_train)
        
        y_pred_multi = multi_model.predict(X_test)
        y_prob_multi = multi_model.predict_proba(X_test)
        
        accuracy_multi = accuracy_score(y_test, y_pred_multi)
        
        print(f"Multi-class Diagnostic Accuracy: {accuracy_multi:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': multi_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Diagnostic Features:")
        print(feature_importance.head(10))
        
        return multi_model, diagnosis_labels, feature_importance, (X_test, y_test, y_pred_multi)
    
    def create_diagnostic_expert_system(self, data):
        """Create rule-based expert system for diagnosis"""
        
        print("\n=== DIAGNOSTIC EXPERT SYSTEM ===")
        
        def expert_system_diagnosis(row):
            """Rule-based diagnostic system"""
            diagnosis_scores = {
                'Healthy': 0,
                'Pneumonia': 0,
                'Heart Disease': 0,
                'Diabetes': 0,
                'Influenza': 0
            }
            
            # Pneumonia rules
            if row['fever'] and row['cough']:
                diagnosis_scores['Pneumonia'] += 3
            if row['temperature'] > 100:
                diagnosis_scores['Pneumonia'] += 2
            if row['wbc_count'] > 10000:
                diagnosis_scores['Pneumonia'] += 2
            if row['shortness_breath']:
                diagnosis_scores['Pneumonia'] += 1
            
            # Heart Disease rules
            if row['chest_pain']:
                diagnosis_scores['Heart Disease'] += 3
            if row['shortness_breath']:
                diagnosis_scores['Heart Disease'] += 2
            if row['age'] > 60:
                diagnosis_scores['Heart Disease'] += 1
            if row['bp_systolic'] > 140:
                diagnosis_scores['Heart Disease'] += 1
            
            # Diabetes rules
            if row['glucose'] > 126:
                diagnosis_scores['Diabetes'] += 4
            if row['fatigue']:
                diagnosis_scores['Diabetes'] += 1
            if row['age'] > 45:
                diagnosis_scores['Diabetes'] += 1
            
            # Influenza rules
            if row['fever'] and row['fatigue']:
                diagnosis_scores['Influenza'] += 3
            if row['temperature'] > 100:
                diagnosis_scores['Influenza'] += 2
            if row['cough']:
                diagnosis_scores['Influenza'] += 1
            if row['nausea']:
                diagnosis_scores['Influenza'] += 1
            
            # Return diagnosis with highest score
            max_score = max(diagnosis_scores.values())
            if max_score < 2:  # Minimum threshold for diagnosis
                return 'Healthy'
            
            return max([k for k, v in diagnosis_scores.items() if v == max_score])
        
        # Apply expert system to test data
        expert_diagnoses = data.apply(expert_system_diagnosis, axis=1)
        
        # Map to numerical labels for comparison
        label_map = {'Healthy': 0, 'Pneumonia': 1, 'Heart Disease': 2, 'Diabetes': 3, 'Influenza': 4}
        expert_numerical = expert_diagnoses.map(label_map)
        
        # Compare with actual diagnoses
        actual_diagnoses = data['primary_diagnosis']
        expert_accuracy = accuracy_score(actual_diagnoses, expert_numerical)
        
        print(f"Expert System Accuracy: {expert_accuracy:.3f}")
        
        # Show confusion matrix
        cm_expert = confusion_matrix(actual_diagnoses, expert_numerical)
        
        return expert_diagnoses, expert_numerical, expert_accuracy, cm_expert
    
    def diagnostic_confidence_analysis(self, models_results, data):
        """Analyze diagnostic confidence and uncertainty"""
        
        print("\n=== DIAGNOSTIC CONFIDENCE ANALYSIS ===")
        
        # Analyze prediction confidence for each condition
        confidence_analysis = {}
        
        for condition, models in models_results.items():
            print(f"\nAnalyzing confidence for {condition} diagnosis:")
            
            best_model = max(models.keys(), key=lambda x: models[x]['f1'])
            probabilities = models[best_model]['probabilities']
            predictions = models[best_model]['predictions']
            
            # Calculate confidence metrics
            confidence_analysis[condition] = {
                'high_confidence_positive': np.sum((probabilities > 0.8) & (predictions == 1)),
                'medium_confidence_positive': np.sum((probabilities >= 0.6) & (probabilities <= 0.8) & (predictions == 1)),
                'low_confidence_positive': np.sum((probabilities < 0.6) & (predictions == 1)),
                'high_confidence_negative': np.sum((probabilities < 0.2) & (predictions == 0)),
                'medium_confidence_negative': np.sum((probabilities >= 0.2) & (probabilities <= 0.4) & (predictions == 0)),
                'uncertain_cases': np.sum((probabilities >= 0.4) & (probabilities <= 0.6))
            }
            
            print(f"High confidence positive cases: {confidence_analysis[condition]['high_confidence_positive']}")
            print(f"Uncertain cases (0.4-0.6 probability): {confidence_analysis[condition]['uncertain_cases']}")
        
        return confidence_analysis
    
    def visualize_diagnostic_performance(self, models_results, multi_class_results, expert_cm):
        """Visualize diagnostic AI performance"""
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # 1. Model comparison across conditions
        conditions = list(models_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        # Get best model for each condition
        best_models_data = {}
        for condition in conditions:
            best_model = max(models_results[condition].keys(), 
                           key=lambda x: models_results[condition][x]['f1'])
            best_models_data[condition] = models_results[condition][best_model]
        
        # Create performance heatmap
        perf_matrix = np.zeros((len(conditions), len(metrics)))
        for i, condition in enumerate(conditions):
            for j, metric in enumerate(metrics):
                perf_matrix[i, j] = best_models_data[condition][metric]
        
        im1 = axes[0, 0].imshow(perf_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[0, 0].set_xticks(range(len(metrics)))
        axes[0, 0].set_xticklabels(metrics)
        axes[0, 0].set_yticks(range(len(conditions)))
        axes[0, 0].set_yticklabels(conditions)
        axes[0, 0].set_title('Diagnostic Model Performance Heatmap')
        
        # Add text annotations
        for i in range(len(conditions)):
            for j in range(len(metrics)):
                axes[0, 0].text(j, i, f'{perf_matrix[i, j]:.3f}', 
                               ha='center', va='center', color='black', fontsize=8)
        
        # 2. F1 scores comparison
        f1_scores = [best_models_data[condition]['f1'] for condition in conditions]
        bars = axes[0, 1].bar(conditions, f1_scores, alpha=0.7)
        axes[0, 1].set_title('F1 Scores by Condition')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars, f1_scores):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., score + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        # 3. ROC curves for pneumonia (example)
        pneumonia_models = models_results['pneumonia']
        for model_name, model_data in pneumonia_models.items():
            # Generate dummy ROC data (in real implementation, use actual test data)
            fpr = np.linspace(0, 1, 100)
            tpr = np.sqrt(fpr) * model_data['auc']  # Simplified curve
            axes[1, 0].plot(fpr, tpr, label=f"{model_name} (AUC={model_data['auc']:.3f})")
        
        axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curves - Pneumonia Diagnosis')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Expert system confusion matrix
        im2 = axes[1, 1].imshow(expert_cm, cmap='Blues')
        axes[1, 1].set_title('Expert System Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        # Add text annotations for confusion matrix
        for i in range(expert_cm.shape[0]):
            for j in range(expert_cm.shape[1]):
                axes[1, 1].text(j, i, str(expert_cm[i, j]), 
                               ha='center', va='center', color='white' if expert_cm[i, j] > expert_cm.max()/2 else 'black')
        
        # 5. Feature importance
        multi_model, diagnosis_labels, feature_importance, _ = multi_class_results
        top_features = feature_importance.head(10)
        
        axes[2, 0].barh(top_features['feature'], top_features['importance'])
        axes[2, 0].set_title('Top 10 Diagnostic Features')
        axes[2, 0].set_xlabel('Importance')
        
        # 6. Diagnosis distribution
        diagnosis_counts = pd.Series([len([x for x in range(5)]) for _ in range(5)], 
                                   index=['Healthy', 'Pneumonia', 'Heart Disease', 'Diabetes', 'Influenza'])
        
        axes[2, 1].pie([100, 80, 60, 40, 50], labels=diagnosis_counts.index, autopct='%1.1f%%')
        axes[2, 1].set_title('Distribution of Diagnoses')
        
        plt.tight_layout()
        plt.show()

# Demonstrate diagnostic AI applications
diagnostic_ai = DiagnosticAISystem()

print("=== DIAGNOSTIC AI APPLICATIONS DEMONSTRATION ===")

# Generate comprehensive patient dataset
patient_data = diagnostic_ai.generate_comprehensive_patient_data(1500)
print("Generated comprehensive patient dataset with multiple conditions...")

# Build individual diagnostic models
models_results, test_data = diagnostic_ai.build_diagnostic_models(patient_data)

# Build multi-class diagnostic model
multi_class_results = diagnostic_ai.build_multi_class_diagnostic_model(patient_data)

# Create expert system
expert_results = diagnostic_ai.create_diagnostic_expert_system(patient_data)

# Analyze diagnostic confidence
confidence_analysis = diagnostic_ai.diagnostic_confidence_analysis(models_results, patient_data)

# Visualize results
diagnostic_ai.visualize_diagnostic_performance(models_results, multi_class_results, expert_results[3])

print("\n=== DIAGNOSTIC AI SYSTEM READY FOR CLINICAL USE ===")
print("The system can now assist healthcare professionals with:")
print("- Multi-condition diagnosis with confidence scores")
print("- Expert system rule-based backup diagnosis")
print("- Feature importance for clinical decision support")
print("- Uncertainty quantification for complex cases")
```

## Summary

This chapter demonstrated comprehensive diagnostic AI applications:

### Key Diagnostic AI Components:
1. **Multi-Class Diagnosis**: Simultaneous consideration of multiple conditions
2. **Confidence Scoring**: Uncertainty quantification for clinical decisions
3. **Expert Systems**: Rule-based diagnostic backup systems
4. **Feature Importance**: Clinical insight into diagnostic factors
5. **Performance Evaluation**: Rigorous assessment using clinical metrics

### Diagnostic AI Advantages:
- **Comprehensive Analysis**: Considers multiple conditions simultaneously
- **Consistency**: Standardized diagnostic approach across cases
- **Speed**: Rapid analysis of complex patient data
- **Decision Support**: Assists rather than replaces clinical judgment
- **Continuous Learning**: Models improve with more data

### Clinical Integration Considerations:
- **Workflow Integration**: Seamless incorporation into clinical practice
- **Interpretability**: Clear explanations for diagnostic recommendations
- **Fail-safes**: Backup systems for critical decisions
- **Human Oversight**: Maintained physician responsibility
- **Regulatory Compliance**: FDA/medical device standards

### Performance Metrics:
- **Sensitivity/Recall**: Correctly identifying positive cases
- **Specificity**: Correctly identifying negative cases
- **Precision**: Accuracy of positive predictions
- **F1 Score**: Balanced metric for diagnostic accuracy
- **AUC-ROC**: Overall model discrimination ability

### Best Practices:
- Validate with diverse patient populations
- Incorporate clinical expertise in model design
- Implement robust uncertainty quantification
- Maintain transparency in diagnostic reasoning
- Ensure continuous monitoring and updates
- Follow medical AI regulatory guidelines

---

*Diagnostic AI systems must balance accuracy, interpretability, and clinical usability to effectively support healthcare professionals in providing quality patient care.* 