# Chapter 35: AI Security Challenges

## Learning Objectives
By the end of this chapter, students will be able to:
- Identify and analyze AI-specific security vulnerabilities
- Implement security frameworks for AI systems
- Design robust defenses against AI attacks
- Apply security testing and validation for AI models
- Understand threat modeling for AI applications

## Table of Contents
1. [Introduction to AI Security](#introduction)
2. [AI Attack Vectors](#attack-vectors)
3. [Model Security Framework](#security-framework)
4. [Threat Detection and Monitoring](#threat-detection)
5. [Security Testing](#security-testing)
6. [Defensive Strategies](#defensive-strategies)

## 1. Introduction to AI Security {#introduction}

AI security encompasses protecting AI systems from various threats including adversarial attacks, data poisoning, model stealing, and privacy breaches.

### AI Security Challenges:
- **Adversarial Examples**: Carefully crafted inputs that fool models
- **Data Poisoning**: Contaminating training data to degrade performance
- **Model Extraction**: Stealing model functionality through queries
- **Privacy Attacks**: Extracting sensitive information from models
- **Backdoor Attacks**: Hidden triggers that cause malicious behavior

### Security Objectives:
- **Confidentiality**: Protect model parameters and training data
- **Integrity**: Ensure model outputs are not maliciously altered
- **Availability**: Maintain system performance under attack
- **Accountability**: Track and audit AI system behavior
- **Fairness**: Prevent security measures from introducing bias

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
import hashlib
import json
import time
from datetime import datetime
warnings.filterwarnings('ignore')

class AISecurityFramework:
    """Comprehensive framework for AI security analysis and defense"""
    
    def __init__(self):
        self.security_logs = []
        self.threat_detectors = {}
        self.defense_mechanisms = {}
        
    def create_security_dataset(self):
        """Create dataset for security demonstrations"""
        # Load digits for realistic security testing
        digits = load_digits()
        X, y = digits.data, digits.target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return (X_train_scaled, X_test_scaled, y_train, y_test), scaler
    
    def generate_adversarial_examples(self, model, X, y, epsilon=0.1):
        """Generate adversarial examples using FGSM (Fast Gradient Sign Method)"""
        print("=== ADVERSARIAL EXAMPLE GENERATION ===")
        
        # Simplified FGSM implementation for demonstration
        adversarial_examples = []
        original_predictions = model.predict(X)
        adversarial_predictions = []
        
        for i, (x, true_label) in enumerate(zip(X[:100], y[:100])):  # Test on subset
            # Get original prediction
            original_pred = model.predict([x])[0]
            
            # Create adversarial perturbation
            # For simplicity, add random noise with controlled magnitude
            noise = np.random.normal(0, epsilon, x.shape)
            adversarial_x = x + noise
            
            # Clip to valid range (assuming normalized data)
            adversarial_x = np.clip(adversarial_x, X.min(), X.max())
            
            # Get adversarial prediction
            adv_pred = model.predict([adversarial_x])[0]
            
            adversarial_examples.append(adversarial_x)
            adversarial_predictions.append(adv_pred)
            
            if i < 10:  # Show first 10 examples
                print(f"Example {i+1}: Original={original_pred}, Adversarial={adv_pred}, "
                      f"Success={'Yes' if original_pred != adv_pred else 'No'}")
        
        # Calculate attack success rate
        original_preds = original_predictions[:100]
        success_rate = np.mean(original_preds != adversarial_predictions)
        
        print(f"\nAdversarial Attack Results:")
        print(f"Attack Success Rate: {success_rate:.2%}")
        print(f"Average Perturbation Magnitude: {epsilon}")
        
        return np.array(adversarial_examples), adversarial_predictions, success_rate
    
    def data_poisoning_simulation(self, X_train, y_train, poison_ratio=0.1):
        """Simulate data poisoning attack"""
        print("\n=== DATA POISONING SIMULATION ===")
        
        n_samples = len(X_train)
        n_poison = int(n_samples * poison_ratio)
        
        # Create poisoned dataset
        X_poisoned = X_train.copy()
        y_poisoned = y_train.copy()
        
        # Random poisoning strategy: flip labels and add noise
        poison_indices = np.random.choice(n_samples, n_poison, replace=False)
        
        for idx in poison_indices:
            # Flip label to random different class
            current_label = y_poisoned[idx]
            possible_labels = [i for i in range(10) if i != current_label]
            y_poisoned[idx] = np.random.choice(possible_labels)
            
            # Add noise to features
            noise = np.random.normal(0, 0.5, X_poisoned[idx].shape)
            X_poisoned[idx] = X_poisoned[idx] + noise
        
        print(f"Poisoned {n_poison} samples ({poison_ratio:.1%} of training data)")
        print(f"Poison indices: {poison_indices[:10]}...")  # Show first 10
        
        return X_poisoned, y_poisoned, poison_indices
    
    def model_extraction_attack(self, target_model, X_test, n_queries=1000):
        """Simulate model extraction attack"""
        print("\n=== MODEL EXTRACTION ATTACK ===")
        
        # Generate query points
        query_data = []
        query_labels = []
        
        # Strategy 1: Random queries within data distribution
        for i in range(n_queries // 2):
            # Sample from normal distribution around test data
            base_sample = X_test[np.random.randint(len(X_test))]
            noise = np.random.normal(0, 0.1, base_sample.shape)
            query_point = base_sample + noise
            
            # Query target model
            pred = target_model.predict([query_point])[0]
            
            query_data.append(query_point)
            query_labels.append(pred)
        
        # Strategy 2: Adversarial queries
        for i in range(n_queries // 2):
            # Use existing test samples with small perturbations
            base_sample = X_test[np.random.randint(len(X_test))]
            perturbation = np.random.uniform(-0.2, 0.2, base_sample.shape)
            query_point = base_sample + perturbation
            
            pred = target_model.predict([query_point])[0]
            
            query_data.append(query_point)
            query_labels.append(pred)
        
        query_data = np.array(query_data)
        query_labels = np.array(query_labels)
        
        # Train surrogate model
        surrogate_model = MLPClassifier(hidden_layer_sizes=(50,), random_state=42, max_iter=500)
        surrogate_model.fit(query_data, query_labels)
        
        # Test fidelity
        test_predictions_target = target_model.predict(X_test)
        test_predictions_surrogate = surrogate_model.predict(X_test)
        
        fidelity = accuracy_score(test_predictions_target, test_predictions_surrogate)
        
        print(f"Model Extraction Results:")
        print(f"Number of queries: {n_queries}")
        print(f"Surrogate model fidelity: {fidelity:.3f}")
        print(f"Attack success: {'High' if fidelity > 0.8 else 'Medium' if fidelity > 0.6 else 'Low'}")
        
        return surrogate_model, fidelity
    
    def privacy_attack_simulation(self, model, X_train, X_test):
        """Simulate membership inference attack"""
        print("\n=== PRIVACY ATTACK SIMULATION ===")
        
        # Get prediction confidence for training and test samples
        train_probs = model.predict_proba(X_train[:500])  # Subset for efficiency
        test_probs = model.predict_proba(X_test[:500])
        
        # Calculate confidence scores (max probability)
        train_confidence = np.max(train_probs, axis=1)
        test_confidence = np.max(test_probs, axis=1)
        
        # Membership inference: training samples tend to have higher confidence
        threshold = np.median(np.concatenate([train_confidence, test_confidence]))
        
        # Classify based on confidence threshold
        train_classified_as_member = train_confidence > threshold
        test_classified_as_member = test_confidence > threshold
        
        # Calculate attack accuracy
        train_accuracy = np.mean(train_classified_as_member)  # Should be high
        test_accuracy = np.mean(~test_classified_as_member)   # Should be high
        
        overall_accuracy = (train_accuracy + test_accuracy) / 2
        
        print(f"Membership Inference Attack Results:")
        print(f"Training samples correctly identified: {train_accuracy:.2%}")
        print(f"Test samples correctly identified: {test_accuracy:.2%}")
        print(f"Overall attack accuracy: {overall_accuracy:.2%}")
        print(f"Privacy risk: {'High' if overall_accuracy > 0.7 else 'Medium' if overall_accuracy > 0.6 else 'Low'}")
        
        return overall_accuracy, threshold
    
    def implement_security_monitoring(self):
        """Implement comprehensive security monitoring system"""
        print("\n=== SECURITY MONITORING SYSTEM ===")
        
        class SecurityMonitor:
            def __init__(self):
                self.log_entries = []
                self.threat_scores = []
                self.alert_threshold = 0.7
                
            def log_prediction(self, input_data, prediction, confidence, metadata=None):
                """Log prediction with security metadata"""
                entry = {
                    'timestamp': datetime.now().isoformat(),
                    'input_hash': hashlib.sha256(str(input_data).encode()).hexdigest()[:16],
                    'prediction': prediction,
                    'confidence': confidence,
                    'metadata': metadata or {}
                }
                self.log_entries.append(entry)
                
                # Calculate threat score
                threat_score = self.calculate_threat_score(input_data, prediction, confidence)
                self.threat_scores.append(threat_score)
                
                # Generate alert if necessary
                if threat_score > self.alert_threshold:
                    self.generate_alert(entry, threat_score)
                
                return entry
            
            def calculate_threat_score(self, input_data, prediction, confidence):
                """Calculate threat score based on multiple indicators"""
                score = 0.0
                
                # Low confidence indicator
                if confidence < 0.5:
                    score += 0.3
                
                # Unusual input pattern (simplified)
                input_mean = np.mean(input_data)
                if input_mean > 2.0 or input_mean < -2.0:  # Assuming normalized data
                    score += 0.4
                
                # High variance in input
                input_var = np.var(input_data)
                if input_var > 1.0:
                    score += 0.3
                
                return min(score, 1.0)
            
            def generate_alert(self, entry, threat_score):
                """Generate security alert"""
                alert = {
                    'timestamp': entry['timestamp'],
                    'alert_type': 'HIGH_THREAT_SCORE',
                    'threat_score': threat_score,
                    'input_hash': entry['input_hash'],
                    'prediction': entry['prediction'],
                    'confidence': entry['confidence']
                }
                print(f"🚨 SECURITY ALERT: Threat score {threat_score:.3f} for input {entry['input_hash']}")
                return alert
            
            def get_security_summary(self):
                """Generate security monitoring summary"""
                if not self.threat_scores:
                    return "No monitoring data available"
                
                avg_threat = np.mean(self.threat_scores)
                max_threat = np.max(self.threat_scores)
                high_threat_count = np.sum(np.array(self.threat_scores) > self.alert_threshold)
                
                summary = {
                    'total_predictions': len(self.threat_scores),
                    'average_threat_score': avg_threat,
                    'maximum_threat_score': max_threat,
                    'high_threat_alerts': high_threat_count,
                    'alert_rate': high_threat_count / len(self.threat_scores) if self.threat_scores else 0
                }
                
                return summary
        
        return SecurityMonitor()
    
    def implement_defensive_measures(self, model, X_train, y_train):
        """Implement various defensive mechanisms"""
        print("\n=== DEFENSIVE MEASURES ===")
        
        class DefensiveFramework:
            def __init__(self, base_model):
                self.base_model = base_model
                self.input_validators = []
                self.output_filters = []
                
            def add_input_validation(self, validator_func):
                """Add input validation mechanism"""
                self.input_validators.append(validator_func)
            
            def add_output_filtering(self, filter_func):
                """Add output filtering mechanism"""
                self.output_filters.append(filter_func)
            
            def secure_predict(self, X):
                """Make predictions with security measures"""
                secure_results = []
                
                for x in X:
                    # Input validation
                    valid_input = True
                    for validator in self.input_validators:
                        if not validator(x):
                            valid_input = False
                            break
                    
                    if not valid_input:
                        # Return safe default or raise exception
                        secure_results.append(-1)  # Invalid input marker
                        continue
                    
                    # Make prediction
                    pred = self.base_model.predict([x])[0]
                    prob = np.max(self.base_model.predict_proba([x])[0])
                    
                    # Output filtering
                    for output_filter in self.output_filters:
                        pred, prob = output_filter(pred, prob)
                    
                    secure_results.append(pred)
                
                return np.array(secure_results)
        
        # Create defensive framework
        defensive_model = DefensiveFramework(model)
        
        # Add input validation
        def range_validator(x):
            """Validate input is within expected range"""
            return np.all(x >= -3) and np.all(x <= 3)  # Assuming normalized data
        
        def variance_validator(x):
            """Validate input variance is reasonable"""
            return np.var(x) < 2.0
        
        defensive_model.add_input_validation(range_validator)
        defensive_model.add_input_validation(variance_validator)
        
        # Add output filtering
        def confidence_filter(pred, prob):
            """Filter low-confidence predictions"""
            if prob < 0.6:
                return -1, prob  # Mark as uncertain
            return pred, prob
        
        defensive_model.add_output_filtering(confidence_filter)
        
        print("Implemented defensive measures:")
        print("- Input range validation")
        print("- Input variance validation")
        print("- Confidence-based output filtering")
        
        return defensive_model
    
    def security_testing_suite(self, model, X_test, y_test):
        """Comprehensive security testing suite"""
        print("\n=== SECURITY TESTING SUITE ===")
        
        test_results = {}
        
        # Test 1: Adversarial robustness
        print("1. Testing adversarial robustness...")
        try:
            _, _, adv_success_rate = self.generate_adversarial_examples(model, X_test[:50], y_test[:50])
            test_results['adversarial_robustness'] = {
                'score': 1 - adv_success_rate,
                'status': 'PASS' if adv_success_rate < 0.3 else 'FAIL'
            }
        except Exception as e:
            test_results['adversarial_robustness'] = {'score': 0, 'status': 'ERROR', 'error': str(e)}
        
        # Test 2: Input validation
        print("2. Testing input validation...")
        invalid_inputs = [
            np.full(X_test[0].shape, 10),  # Out of range
            np.full(X_test[0].shape, -10), # Out of range
            np.full(X_test[0].shape, np.nan), # NaN values
        ]
        
        validation_passed = 0
        for invalid_input in invalid_inputs:
            try:
                pred = model.predict([invalid_input])
                # If prediction succeeds, input validation might be weak
            except:
                validation_passed += 1
        
        test_results['input_validation'] = {
            'score': validation_passed / len(invalid_inputs),
            'status': 'PASS' if validation_passed >= 2 else 'FAIL'
        }
        
        # Test 3: Model confidence calibration
        print("3. Testing confidence calibration...")
        probs = model.predict_proba(X_test[:100])
        max_probs = np.max(probs, axis=1)
        preds = model.predict(X_test[:100])
        correct = (preds == y_test[:100])
        
        # Check if high confidence correlates with correctness
        high_conf_mask = max_probs > 0.8
        if np.sum(high_conf_mask) > 0:
            high_conf_accuracy = np.mean(correct[high_conf_mask])
            calibration_score = high_conf_accuracy
        else:
            calibration_score = 0.5
        
        test_results['confidence_calibration'] = {
            'score': calibration_score,
            'status': 'PASS' if calibration_score > 0.8 else 'FAIL'
        }
        
        # Test 4: Consistency under perturbations
        print("4. Testing prediction consistency...")
        base_predictions = model.predict(X_test[:50])
        consistency_scores = []
        
        for i, x in enumerate(X_test[:50]):
            # Add small perturbations
            perturbed_predictions = []
            for _ in range(10):
                noise = np.random.normal(0, 0.01, x.shape)
                perturbed_x = x + noise
                pred = model.predict([perturbed_x])[0]
                perturbed_predictions.append(pred)
            
            # Calculate consistency
            consistency = np.mean(np.array(perturbed_predictions) == base_predictions[i])
            consistency_scores.append(consistency)
        
        avg_consistency = np.mean(consistency_scores)
        test_results['prediction_consistency'] = {
            'score': avg_consistency,
            'status': 'PASS' if avg_consistency > 0.9 else 'FAIL'
        }
        
        # Generate overall security score
        scores = [result['score'] for result in test_results.values() if 'score' in result]
        overall_score = np.mean(scores) if scores else 0
        
        print(f"\n=== SECURITY TEST RESULTS ===")
        for test_name, result in test_results.items():
            print(f"{test_name}: {result['status']} (Score: {result['score']:.3f})")
        
        print(f"\nOverall Security Score: {overall_score:.3f}")
        security_level = 'HIGH' if overall_score > 0.8 else 'MEDIUM' if overall_score > 0.6 else 'LOW'
        print(f"Security Level: {security_level}")
        
        return test_results, overall_score
    
    def visualize_security_analysis(self, adv_examples, poison_data, threat_scores, test_results):
        """Visualize security analysis results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Adversarial examples visualization (first few)
        if len(adv_examples) > 0:
            for i in range(min(6, len(adv_examples))):
                ax = axes[0, i % 3] if i < 3 else axes[1, i % 3]
                img = adv_examples[i].reshape(8, 8)  # Assuming 8x8 digit images
                ax.imshow(img, cmap='gray')
                ax.set_title(f'Adversarial {i+1}')
                ax.axis('off')
        
        # Fill remaining subplots with analysis
        if len(adv_examples) < 6:
            # Threat score distribution
            if threat_scores:
                axes[0, 2].hist(threat_scores, bins=20, alpha=0.7, color='red')
                axes[0, 2].axvline(0.7, color='black', linestyle='--', label='Alert Threshold')
                axes[0, 2].set_title('Threat Score Distribution')
                axes[0, 2].set_xlabel('Threat Score')
                axes[0, 2].set_ylabel('Frequency')
                axes[0, 2].legend()
            
            # Security test results
            if test_results:
                test_names = list(test_results.keys())
                test_scores = [test_results[name]['score'] for name in test_names]
                
                axes[1, 0].bar(range(len(test_names)), test_scores, alpha=0.7)
                axes[1, 0].set_xticks(range(len(test_names)))
                axes[1, 0].set_xticklabels([name.replace('_', ' ').title() for name in test_names], 
                                          rotation=45, ha='right')
                axes[1, 0].set_title('Security Test Scores')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].set_ylim(0, 1)
            
            # Attack success rates comparison
            attack_types = ['Adversarial', 'Data Poisoning', 'Model Extraction', 'Privacy Attack']
            # Mock success rates for visualization
            success_rates = [0.4, 0.6, 0.7, 0.5]
            
            axes[1, 1].bar(attack_types, success_rates, alpha=0.7, color=['red', 'orange', 'yellow', 'pink'])
            axes[1, 1].set_title('Attack Success Rates')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Security timeline (mock data)
            times = range(24)  # 24 hours
            security_incidents = np.random.poisson(2, 24)  # Mock incident counts
            
            axes[1, 2].plot(times, security_incidents, marker='o')
            axes[1, 2].set_title('Security Incidents Over Time')
            axes[1, 2].set_xlabel('Hour')
            axes[1, 2].set_ylabel('Incident Count')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Demonstrate AI security framework
security_framework = AISecurityFramework()

print("=== AI SECURITY CHALLENGES AND DEFENSES ===")

# Create secure dataset
(X_train, X_test, y_train, y_test), scaler = security_framework.create_security_dataset()
print("Dataset prepared for security analysis")

# Train baseline model
model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
model.fit(X_train, y_train)

baseline_accuracy = model.score(X_test, y_test)
print(f"Baseline model accuracy: {baseline_accuracy:.3f}")

# Security Analysis

# 1. Adversarial examples
adv_examples, adv_preds, adv_success = security_framework.generate_adversarial_examples(
    model, X_test, y_test
)

# 2. Data poisoning
X_poisoned, y_poisoned, poison_indices = security_framework.data_poisoning_simulation(
    X_train, y_train, poison_ratio=0.1
)

# Train model on poisoned data
poisoned_model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
poisoned_model.fit(X_poisoned, y_poisoned)
poisoned_accuracy = poisoned_model.score(X_test, y_test)
print(f"Poisoned model accuracy: {poisoned_accuracy:.3f} (drop: {baseline_accuracy - poisoned_accuracy:.3f})")

# 3. Model extraction
surrogate_model, fidelity = security_framework.model_extraction_attack(model, X_test)

# 4. Privacy attack
privacy_accuracy, threshold = security_framework.privacy_attack_simulation(model, X_train, X_test)

# 5. Security monitoring
monitor = security_framework.implement_security_monitoring()

# Test monitoring with some samples
threat_scores = []
for i in range(50):
    sample = X_test[i]
    pred = model.predict([sample])[0]
    prob = np.max(model.predict_proba([sample])[0])
    
    entry = monitor.log_prediction(sample, pred, prob)
    threat_scores.append(monitor.threat_scores[-1])

security_summary = monitor.get_security_summary()
print(f"\nSecurity Monitoring Summary:")
for key, value in security_summary.items():
    print(f"  {key}: {value}")

# 6. Defensive measures
defensive_model = security_framework.implement_defensive_measures(model, X_train, y_train)

# Test defensive model
defensive_predictions = defensive_model.secure_predict(X_test[:20])
valid_predictions = defensive_predictions[defensive_predictions != -1]
print(f"Defensive model: {len(valid_predictions)}/{len(defensive_predictions)} predictions accepted")

# 7. Security testing
test_results, overall_score = security_framework.security_testing_suite(model, X_test, y_test)

# Visualize results
security_framework.visualize_security_analysis(
    adv_examples[:6], (X_poisoned, poison_indices), threat_scores, test_results
)
```

## Summary

This chapter covered comprehensive AI security challenges and defense mechanisms:

### Key Security Threats:
1. **Adversarial Attacks**: Carefully crafted inputs that fool models
2. **Data Poisoning**: Contaminating training data to degrade performance
3. **Model Extraction**: Stealing model functionality through queries
4. **Privacy Attacks**: Extracting sensitive information from models
5. **Backdoor Attacks**: Hidden triggers causing malicious behavior

### Defense Strategies:
- **Input Validation**: Robust preprocessing and anomaly detection
- **Adversarial Training**: Training models with adversarial examples
- **Model Monitoring**: Continuous threat detection and logging
- **Output Filtering**: Confidence-based prediction filtering
- **Privacy Preservation**: Differential privacy and federated learning

### Security Best Practices:
- Implement comprehensive security testing frameworks
- Monitor model behavior continuously in production
- Use defense-in-depth strategies with multiple security layers
- Regular security audits and penetration testing
- Stay updated on emerging AI security threats and defenses

---

## Exercises

1. **Custom Attack**: Implement a specific adversarial attack method
2. **Defense Mechanism**: Design robust input validation system
3. **Monitoring Dashboard**: Build real-time security monitoring interface
4. **Privacy Framework**: Implement differential privacy mechanisms
5. **Security Audit**: Conduct comprehensive security assessment

---

*AI security requires proactive defense measures and continuous monitoring to protect against evolving threats while maintaining model performance and utility.* 