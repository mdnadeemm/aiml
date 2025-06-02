# Chapter 31: AI for Patient Care and Monitoring

## Learning Objectives
By the end of this chapter, students will be able to:
- Implement AI systems for continuous patient monitoring
- Design wearable device data analysis and alert systems
- Apply machine learning to electronic health records
- Build predictive models for patient deterioration and readmission
- Develop AI-powered patient engagement and adherence tools

## Table of Contents
1. [Introduction to AI Patient Monitoring](#introduction)
2. [Wearable Devices and IoT Healthcare](#wearables)
3. [Electronic Health Records Analysis](#ehr-analysis)
4. [Predictive Analytics for Patient Outcomes](#predictive-analytics)
5. [Remote Patient Monitoring Systems](#remote-monitoring)
6. [AI-Powered Patient Engagement](#patient-engagement)

## 1. Introduction to AI Patient Monitoring {#introduction}

AI-powered patient care and monitoring systems enable continuous, real-time assessment of patient health, early detection of complications, and personalized care delivery.

### Key Applications:
- **Continuous Monitoring**: Real-time vital sign tracking and analysis
- **Early Warning Systems**: Predicting patient deterioration before crisis
- **Medication Adherence**: Tracking and improving treatment compliance
- **Remote Care**: Managing patients outside traditional healthcare settings
- **Personalized Care Plans**: Tailoring interventions to individual needs

### Benefits:
- **Proactive Care**: Early intervention before complications
- **Resource Optimization**: Efficient allocation of healthcare resources
- **Patient Safety**: Reduced medical errors and adverse events
- **Cost Reduction**: Preventing expensive emergency interventions
- **Accessibility**: Care delivery to remote or underserved populations

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PatientMonitoringSystem:
    """Comprehensive AI system for patient care and monitoring"""
    
    def __init__(self):
        self.models = {}
        self.monitoring_data = {}
        self.alerts = []
        
    def generate_patient_monitoring_data(self, n_patients=100, days=30):
        """Generate synthetic patient monitoring data"""
        np.random.seed(42)
        
        patients_data = []
        
        for patient_id in range(n_patients):
            # Patient characteristics
            age = np.random.normal(65, 15)
            age = max(18, min(95, age))
            
            gender = np.random.choice(['M', 'F'])
            
            # Chronic conditions
            diabetes = np.random.binomial(1, 0.3)
            hypertension = np.random.binomial(1, 0.4)
            heart_disease = np.random.binomial(1, 0.2)
            
            # Risk factors
            baseline_risk = age * 0.01 + diabetes * 0.2 + hypertension * 0.15 + heart_disease * 0.25
            
            # Generate daily monitoring data
            for day in range(days):
                # Simulate day-to-day variation
                daily_variation = np.random.normal(0, 0.1)
                current_risk = baseline_risk + daily_variation
                
                # Vital signs with some correlation to risk
                heart_rate = np.random.normal(75 + current_risk * 20, 10)
                heart_rate = max(50, min(120, heart_rate))
                
                blood_pressure_sys = np.random.normal(130 + current_risk * 30, 15)
                blood_pressure_sys = max(90, min(200, blood_pressure_sys))
                
                blood_pressure_dia = np.random.normal(80 + current_risk * 15, 10)
                blood_pressure_dia = max(60, min(120, blood_pressure_dia))
                
                temperature = np.random.normal(98.6 + current_risk * 2, 0.5)
                oxygen_saturation = np.random.normal(98 - current_risk * 5, 2)
                oxygen_saturation = max(85, min(100, oxygen_saturation))
                
                # Activity levels
                steps = np.random.poisson(5000 - current_risk * 2000)
                steps = max(0, steps)
                
                sleep_hours = np.random.normal(7 - current_risk * 2, 1)
                sleep_hours = max(3, min(12, sleep_hours))
                
                # Medication adherence
                med_adherence = np.random.binomial(1, 0.8 - current_risk * 0.3)
                
                # Critical events (rare but important)
                critical_event = np.random.binomial(1, current_risk * 0.05)
                
                patients_data.append({
                    'patient_id': patient_id,
                    'day': day,
                    'age': age,
                    'gender': gender,
                    'diabetes': diabetes,
                    'hypertension': hypertension,
                    'heart_disease': heart_disease,
                    'heart_rate': heart_rate,
                    'bp_systolic': blood_pressure_sys,
                    'bp_diastolic': blood_pressure_dia,
                    'temperature': temperature,
                    'oxygen_saturation': oxygen_saturation,
                    'steps': steps,
                    'sleep_hours': sleep_hours,
                    'med_adherence': med_adherence,
                    'baseline_risk': baseline_risk,
                    'current_risk': current_risk,
                    'critical_event': critical_event
                })
        
        return pd.DataFrame(patients_data)
    
    def implement_early_warning_system(self, monitoring_data):
        """Implement early warning system for patient deterioration"""
        print("=== EARLY WARNING SYSTEM IMPLEMENTATION ===")
        
        # Prepare features for early warning
        features = ['age', 'diabetes', 'hypertension', 'heart_disease',
                   'heart_rate', 'bp_systolic', 'bp_diastolic', 'temperature',
                   'oxygen_saturation', 'steps', 'sleep_hours', 'med_adherence']
        
        X = monitoring_data[features]
        y = monitoring_data['critical_event']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train early warning model
        early_warning_model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        )
        early_warning_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = early_warning_model.predict(X_test)
        y_prob = early_warning_model.predict_proba(X_test)[:, 1]
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_prob)
        
        print(f"Early Warning System Performance:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"AUC Score: {auc_score:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': early_warning_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nMost Important Warning Indicators:")
        print(feature_importance.head())
        
        # Generate risk scores for all patients
        risk_scores = early_warning_model.predict_proba(X)[:, 1]
        monitoring_data = monitoring_data.copy()
        monitoring_data['risk_score'] = risk_scores
        
        return early_warning_model, monitoring_data, feature_importance
    
    def anomaly_detection_system(self, monitoring_data):
        """Implement anomaly detection for unusual patient patterns"""
        print("\n=== ANOMALY DETECTION SYSTEM ===")
        
        # Use vital signs for anomaly detection
        vital_features = ['heart_rate', 'bp_systolic', 'bp_diastolic', 
                         'temperature', 'oxygen_saturation']
        
        X_vitals = monitoring_data[vital_features]
        
        # Normalize features
        scaler = StandardScaler()
        X_vitals_scaled = scaler.fit_transform(X_vitals)
        
        # Train isolation forest for anomaly detection
        anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        anomalies = anomaly_detector.fit_predict(X_vitals_scaled)
        
        # Convert to binary (1 = normal, 0 = anomaly)
        monitoring_data = monitoring_data.copy()
        monitoring_data['anomaly'] = (anomalies == -1).astype(int)
        
        print(f"Detected {monitoring_data['anomaly'].sum()} anomalous readings")
        print(f"Anomaly rate: {monitoring_data['anomaly'].mean()*100:.1f}%")
        
        return anomaly_detector, monitoring_data, scaler
    
    def medication_adherence_tracking(self, monitoring_data):
        """Track and predict medication adherence patterns"""
        print("\n=== MEDICATION ADHERENCE TRACKING ===")
        
        # Analyze adherence patterns by patient
        adherence_analysis = monitoring_data.groupby('patient_id').agg({
            'med_adherence': ['mean', 'std', 'count'],
            'critical_event': 'sum',
            'diabetes': 'first',
            'hypertension': 'first',
            'heart_disease': 'first',
            'age': 'first'
        }).round(3)
        
        adherence_analysis.columns = ['_'.join(col).strip() for col in adherence_analysis.columns]
        adherence_analysis = adherence_analysis.reset_index()
        
        # Categorize adherence levels
        adherence_analysis['adherence_category'] = pd.cut(
            adherence_analysis['med_adherence_mean'],
            bins=[0, 0.6, 0.8, 1.0],
            labels=['Poor', 'Moderate', 'Good']
        )
        
        print("Medication Adherence Distribution:")
        print(adherence_analysis['adherence_category'].value_counts())
        
        # Predict non-adherence risk
        features_adherence = ['diabetes_first', 'hypertension_first', 
                             'heart_disease_first', 'age_first']
        
        # Create binary target for poor adherence
        poor_adherence = (adherence_analysis['med_adherence_mean'] < 0.8).astype(int)
        
        X_adh = adherence_analysis[features_adherence]
        
        # Train adherence prediction model
        adherence_model = LogisticRegression(random_state=42)
        adherence_model.fit(X_adh, poor_adherence)
        
        adherence_risk = adherence_model.predict_proba(X_adh)[:, 1]
        adherence_analysis['adherence_risk'] = adherence_risk
        
        print(f"\nPatients at high risk for non-adherence: {np.sum(adherence_risk > 0.5)}")
        
        return adherence_model, adherence_analysis
    
    def remote_monitoring_dashboard(self, monitoring_data, patient_id=0):
        """Create a remote monitoring dashboard for a specific patient"""
        print(f"\n=== REMOTE MONITORING DASHBOARD - PATIENT {patient_id} ===")
        
        # Filter data for specific patient
        patient_data = monitoring_data[monitoring_data['patient_id'] == patient_id].copy()
        patient_data = patient_data.sort_values('day')
        
        if len(patient_data) == 0:
            print(f"No data found for patient {patient_id}")
            return
        
        # Patient summary
        print(f"Patient Demographics:")
        print(f"Age: {patient_data['age'].iloc[0]:.0f}")
        print(f"Gender: {patient_data['gender'].iloc[0]}")
        print(f"Chronic Conditions: ", end="")
        conditions = []
        if patient_data['diabetes'].iloc[0]: conditions.append("Diabetes")
        if patient_data['hypertension'].iloc[0]: conditions.append("Hypertension")  
        if patient_data['heart_disease'].iloc[0]: conditions.append("Heart Disease")
        print(", ".join(conditions) if conditions else "None")
        
        # Recent vital signs (last 7 days)
        recent_data = patient_data.tail(7)
        
        print(f"\nRecent Vital Signs (Last 7 Days):")
        print(f"Average Heart Rate: {recent_data['heart_rate'].mean():.0f} bpm")
        print(f"Average Blood Pressure: {recent_data['bp_systolic'].mean():.0f}/{recent_data['bp_diastolic'].mean():.0f}")
        print(f"Average Temperature: {recent_data['temperature'].mean():.1f}°F")
        print(f"Average Oxygen Saturation: {recent_data['oxygen_saturation'].mean():.1f}%")
        
        # Activity summary
        print(f"\nActivity Summary:")
        print(f"Average Daily Steps: {recent_data['steps'].mean():.0f}")
        print(f"Average Sleep: {recent_data['sleep_hours'].mean():.1f} hours")
        print(f"Medication Adherence: {recent_data['med_adherence'].mean()*100:.0f}%")
        
        # Risk assessment
        avg_risk = recent_data['risk_score'].mean() if 'risk_score' in recent_data.columns else 0
        print(f"\nRisk Assessment:")
        print(f"Current Risk Score: {avg_risk:.3f}")
        
        if avg_risk > 0.7:
            print("⚠️  HIGH RISK - Immediate attention recommended")
        elif avg_risk > 0.4:
            print("⚡ MODERATE RISK - Monitor closely")
        else:
            print("✅ LOW RISK - Continue routine monitoring")
        
        # Recent anomalies
        if 'anomaly' in recent_data.columns:
            anomaly_count = recent_data['anomaly'].sum()
            if anomaly_count > 0:
                print(f"\n🔔 {anomaly_count} anomalous readings in last 7 days")
        
        return patient_data
    
    def generate_care_recommendations(self, patient_data, adherence_analysis):
        """Generate AI-powered care recommendations"""
        print("\n=== AI-POWERED CARE RECOMMENDATIONS ===")
        
        recommendations = {}
        
        # Analyze each patient
        for patient_id in patient_data['patient_id'].unique()[:5]:  # Show first 5 patients
            patient_records = patient_data[patient_data['patient_id'] == patient_id]
            recent_data = patient_records.tail(7)  # Last week
            
            patient_recommendations = []
            
            # Risk-based recommendations
            if 'risk_score' in recent_data.columns:
                avg_risk = recent_data['risk_score'].mean()
                if avg_risk > 0.6:
                    patient_recommendations.append("Schedule immediate physician consultation")
                    patient_recommendations.append("Increase monitoring frequency to twice daily")
                elif avg_risk > 0.3:
                    patient_recommendations.append("Schedule follow-up within 1 week")
                    patient_recommendations.append("Monitor vital signs daily")
            
            # Vital sign recommendations
            avg_hr = recent_data['heart_rate'].mean()
            if avg_hr > 100:
                patient_recommendations.append("Heart rate elevated - consider cardiology referral")
            elif avg_hr < 60:
                patient_recommendations.append("Bradycardia detected - evaluate medications")
            
            avg_bp_sys = recent_data['bp_systolic'].mean()
            if avg_bp_sys > 140:
                patient_recommendations.append("Hypertension management needed - adjust medications")
            elif avg_bp_sys < 90:
                patient_recommendations.append("Hypotension risk - review medications")
            
            avg_spo2 = recent_data['oxygen_saturation'].mean()
            if avg_spo2 < 95:
                patient_recommendations.append("Oxygen saturation low - pulmonary evaluation needed")
            
            # Activity recommendations
            avg_steps = recent_data['steps'].mean()
            if avg_steps < 2000:
                patient_recommendations.append("Low activity level - physical therapy consultation")
            elif avg_steps > 8000:
                patient_recommendations.append("Excellent activity level - continue current regimen")
            
            # Sleep recommendations
            avg_sleep = recent_data['sleep_hours'].mean()
            if avg_sleep < 6:
                patient_recommendations.append("Insufficient sleep - sleep study recommended")
            elif avg_sleep > 10:
                patient_recommendations.append("Excessive sleep - evaluate for depression/sleep disorders")
            
            # Medication adherence
            adherence_rate = recent_data['med_adherence'].mean()
            if adherence_rate < 0.8:
                patient_recommendations.append("Poor medication adherence - medication management program")
                patient_recommendations.append("Consider medication reminders or simplified regimen")
            
            # Anomaly-based recommendations
            if 'anomaly' in recent_data.columns and recent_data['anomaly'].sum() > 2:
                patient_recommendations.append("Multiple anomalous readings - comprehensive evaluation needed")
            
            recommendations[patient_id] = patient_recommendations
        
        # Display recommendations
        for patient_id, recs in recommendations.items():
            print(f"\nPatient {patient_id} Recommendations:")
            for i, rec in enumerate(recs, 1):
                print(f"  {i}. {rec}")
        
        return recommendations
    
    def visualize_monitoring_insights(self, monitoring_data, feature_importance):
        """Visualize patient monitoring insights"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Risk score distribution
        if 'risk_score' in monitoring_data.columns:
            axes[0, 0].hist(monitoring_data['risk_score'], bins=30, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(monitoring_data['risk_score'].mean(), color='red', 
                              linestyle='--', label='Mean Risk')
            axes[0, 0].set_title('Patient Risk Score Distribution')
            axes[0, 0].set_xlabel('Risk Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
        
        # 2. Vital signs correlation
        vital_signs = ['heart_rate', 'bp_systolic', 'oxygen_saturation', 'temperature']
        correlation_matrix = monitoring_data[vital_signs].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 1])
        axes[0, 1].set_title('Vital Signs Correlation Matrix')
        
        # 3. Feature importance
        axes[0, 2].barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
        axes[0, 2].set_title('Top 10 Early Warning Indicators')
        axes[0, 2].set_xlabel('Importance')
        
        # 4. Critical events by condition
        condition_events = monitoring_data.groupby(['diabetes', 'hypertension', 'heart_disease'])['critical_event'].mean().reset_index()
        condition_labels = []
        for _, row in condition_events.iterrows():
            label = []
            if row['diabetes']: label.append('DM')
            if row['hypertension']: label.append('HTN')
            if row['heart_disease']: label.append('HD')
            condition_labels.append('+'.join(label) if label else 'None')
        
        axes[1, 0].bar(range(len(condition_labels)), condition_events['critical_event'], alpha=0.7)
        axes[1, 0].set_title('Critical Event Rate by Conditions')
        axes[1, 0].set_xlabel('Condition Combination')
        axes[1, 0].set_ylabel('Critical Event Rate')
        axes[1, 0].set_xticks(range(len(condition_labels)))
        axes[1, 0].set_xticklabels(condition_labels, rotation=45)
        
        # 5. Activity levels vs outcomes
        monitoring_data['activity_level'] = pd.cut(monitoring_data['steps'], 
                                                  bins=[0, 2000, 5000, float('inf')], 
                                                  labels=['Low', 'Moderate', 'High'])
        
        activity_outcomes = monitoring_data.groupby('activity_level')['critical_event'].mean()
        axes[1, 1].bar(activity_outcomes.index, activity_outcomes.values, alpha=0.7)
        axes[1, 1].set_title('Critical Events by Activity Level')
        axes[1, 1].set_ylabel('Critical Event Rate')
        
        # 6. Medication adherence impact
        adherence_bins = pd.cut(monitoring_data['med_adherence'], 
                               bins=[0, 0.6, 0.8, 1.0], 
                               labels=['Poor', 'Moderate', 'Good'])
        adherence_outcomes = monitoring_data.groupby(adherence_bins)['critical_event'].mean()
        
        axes[1, 2].bar(adherence_outcomes.index, adherence_outcomes.values, alpha=0.7)
        axes[1, 2].set_title('Critical Events by Medication Adherence')
        axes[1, 2].set_ylabel('Critical Event Rate')
        
        plt.tight_layout()
        plt.show()

# Demonstrate patient monitoring system
monitoring_system = PatientMonitoringSystem()

print("=== AI FOR PATIENT CARE AND MONITORING ===")

# Generate monitoring data
monitoring_data = monitoring_system.generate_patient_monitoring_data(n_patients=100, days=30)
print(f"Generated monitoring data for {monitoring_data['patient_id'].nunique()} patients over {monitoring_data['day'].nunique()} days")

# Implement early warning system
early_warning_model, enhanced_data, feature_importance = monitoring_system.implement_early_warning_system(monitoring_data)

# Anomaly detection
anomaly_detector, enhanced_data, scaler = monitoring_system.anomaly_detection_system(enhanced_data)

# Medication adherence tracking
adherence_model, adherence_analysis = monitoring_system.medication_adherence_tracking(enhanced_data)

# Remote monitoring dashboard
patient_dashboard = monitoring_system.remote_monitoring_dashboard(enhanced_data, patient_id=0)

# Generate care recommendations
recommendations = monitoring_system.generate_care_recommendations(enhanced_data, adherence_analysis)

# Visualize insights
monitoring_system.visualize_monitoring_insights(enhanced_data, feature_importance)
```

## Summary

This chapter demonstrated AI applications in patient care and monitoring:

### Key Technologies:
1. **Early Warning Systems**: Predicting patient deterioration
2. **Anomaly Detection**: Identifying unusual patterns in vital signs
3. **Medication Adherence**: Tracking and improving compliance
4. **Remote Monitoring**: Continuous care outside hospital settings
5. **Predictive Analytics**: Forecasting patient outcomes

### Implementation Benefits:
- **Proactive Care**: Early intervention before crises
- **Personalized Monitoring**: Tailored to individual patient needs
- **Resource Optimization**: Efficient allocation of healthcare resources
- **Improved Outcomes**: Better patient safety and care quality
- **Cost Reduction**: Preventing expensive emergency interventions

### Best Practices:
- Integrate multiple data sources for comprehensive monitoring
- Implement robust alert systems with appropriate thresholds
- Ensure data privacy and security compliance
- Provide clinician training on AI system interpretation
- Continuously validate and update monitoring algorithms
- Design user-friendly interfaces for patients and providers

---

## Exercises

1. **Wearable Integration**: Design system to integrate multiple wearable devices
2. **Alert Optimization**: Optimize alert thresholds to minimize false positives
3. **Patient Engagement**: Build mobile app for patient self-monitoring
4. **Predictive Modeling**: Develop models for specific disease progression
5. **Real-time Processing**: Implement streaming data analysis for real-time monitoring

---

*AI-powered patient monitoring systems enable continuous, personalized care that improves outcomes while reducing costs and enhancing patient safety.* 