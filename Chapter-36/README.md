# Chapter 36: Protecting AI from Adversarial Attacks

## Learning Objectives
By the end of this chapter, students will be able to:
- Understand different types of adversarial attacks and their mechanisms
- Implement robust defense strategies against adversarial examples
- Apply adversarial training and defensive distillation techniques
- Design detection systems for adversarial inputs
- Evaluate adversarial robustness of AI models

## Table of Contents
1. [Understanding Adversarial Attacks](#understanding-attacks)
2. [Defense Mechanisms](#defense-mechanisms)
3. [Adversarial Training](#adversarial-training)
4. [Detection and Filtering](#detection-filtering)
5. [Robustness Evaluation](#robustness-evaluation)
6. [Advanced Defense Strategies](#advanced-defense)

## 1. Understanding Adversarial Attacks {#understanding-attacks}

Adversarial attacks exploit vulnerabilities in AI models by creating inputs that are imperceptibly different from normal inputs but cause the model to make incorrect predictions.

### Types of Adversarial Attacks:
- **FGSM (Fast Gradient Sign Method)**: Single-step attack using gradient signs
- **PGD (Projected Gradient Descent)**: Multi-step iterative attack
- **C&W (Carlini & Wagner)**: Optimization-based attack with different norms
- **DeepFool**: Minimal perturbation attack to cross decision boundaries
- **Black-box Attacks**: Attacks without access to model parameters

### Attack Properties:
- **Perturbation Budget**: Maximum allowed distortion (ε)
- **Norm Constraints**: L0, L2, L∞ norm limitations
- **Transferability**: Attacks crafted on one model affecting others
- **Targeted vs Untargeted**: Specific target class vs any misclassification

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

class AdversarialDefenseFramework:
    """Comprehensive framework for adversarial attack defense"""
    
    def __init__(self):
        self.attack_history = []
        self.defense_results = {}
        
    def create_neural_network_model(self, input_shape, num_classes):
        """Create a simple neural network for adversarial testing"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def generate_fgsm_attack(self, model, x, y_true, epsilon=0.1):
        """Generate FGSM (Fast Gradient Sign Method) adversarial examples"""
        print("=== FGSM ADVERSARIAL ATTACK ===")
        
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y_true, dtype=tf.int64)
        
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            predictions = model(x_tensor)
            loss = keras.losses.sparse_categorical_crossentropy(y_tensor, predictions)
        
        # Get gradients
        gradients = tape.gradient(loss, x_tensor)
        
        # Create adversarial examples
        signed_grad = tf.sign(gradients)
        adversarial_x = x_tensor + epsilon * signed_grad
        
        # Clip to valid range [0, 1]
        adversarial_x = tf.clip_by_value(adversarial_x, 0, 1)
        
        return adversarial_x.numpy()
    
    def generate_pgd_attack(self, model, x, y_true, epsilon=0.1, alpha=0.01, num_iter=10):
        """Generate PGD (Projected Gradient Descent) adversarial examples"""
        print("=== PGD ADVERSARIAL ATTACK ===")
        
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y_true, dtype=tf.int64)
        
        # Initialize adversarial example
        adversarial_x = x_tensor + tf.random.uniform(tf.shape(x_tensor), -epsilon, epsilon)
        adversarial_x = tf.clip_by_value(adversarial_x, 0, 1)
        
        for i in range(num_iter):
            with tf.GradientTape() as tape:
                tape.watch(adversarial_x)
                predictions = model(adversarial_x)
                loss = keras.losses.sparse_categorical_crossentropy(y_tensor, predictions)
            
            gradients = tape.gradient(loss, adversarial_x)
            signed_grad = tf.sign(gradients)
            
            # Update adversarial example
            adversarial_x = adversarial_x + alpha * signed_grad
            
            # Project back to L∞ ball
            perturbation = adversarial_x - x_tensor
            perturbation = tf.clip_by_value(perturbation, -epsilon, epsilon)
            adversarial_x = x_tensor + perturbation
            adversarial_x = tf.clip_by_value(adversarial_x, 0, 1)
        
        return adversarial_x.numpy()
    
    def implement_adversarial_training(self, X_train, y_train, X_test, y_test, 
                                     input_shape, num_classes, epsilon=0.1):
        """Implement adversarial training defense"""
        print("=== ADVERSARIAL TRAINING DEFENSE ===")
        
        # Create and train standard model
        standard_model = self.create_neural_network_model(input_shape, num_classes)
        print("Training standard model...")
        standard_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.2)
        
        # Create adversarial training model
        adversarial_model = self.create_neural_network_model(input_shape, num_classes)
        
        print("Training adversarial model with mixed clean and adversarial examples...")
        
        # Custom training loop for adversarial training
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        loss_fn = keras.losses.SparseCategoricalCrossentropy()
        
        # Training parameters
        epochs = 10
        batch_size = 32
        num_batches = len(X_train) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_x = X_train[start_idx:end_idx]
                batch_y = y_train[start_idx:end_idx]
                
                # Generate adversarial examples for half the batch
                half_size = batch_size // 2
                clean_x = batch_x[:half_size]
                clean_y = batch_y[:half_size]
                
                adv_source_x = batch_x[half_size:]
                adv_source_y = batch_y[half_size:]
                
                # Generate adversarial examples
                adv_x = self.generate_fgsm_attack(adversarial_model, adv_source_x, adv_source_y, epsilon)
                
                # Combine clean and adversarial examples
                mixed_x = np.vstack([clean_x, adv_x])
                mixed_y = np.hstack([clean_y, adv_source_y])
                
                # Training step
                with tf.GradientTape() as tape:
                    predictions = adversarial_model(mixed_x, training=True)
                    loss = loss_fn(mixed_y, predictions)
                
                gradients = tape.gradient(loss, adversarial_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, adversarial_model.trainable_variables))
                
                # Calculate metrics
                epoch_loss += loss.numpy()
                epoch_acc += accuracy_score(mixed_y, np.argmax(predictions.numpy(), axis=1))
            
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
        
        # Evaluate both models
        print("\\nEvaluating models...")
        
        # Standard evaluation
        standard_acc = standard_model.evaluate(X_test, y_test, verbose=0)[1]
        adversarial_acc = adversarial_model.evaluate(X_test, y_test, verbose=0)[1]
        
        print(f"Standard model clean accuracy: {standard_acc:.4f}")
        print(f"Adversarial model clean accuracy: {adversarial_acc:.4f}")
        
        # Adversarial evaluation
        test_adv_x = self.generate_fgsm_attack(standard_model, X_test[:100], y_test[:100], epsilon)
        
        standard_adv_acc = accuracy_score(y_test[:100], 
                                        np.argmax(standard_model.predict(test_adv_x, verbose=0), axis=1))
        adversarial_adv_acc = accuracy_score(y_test[:100], 
                                           np.argmax(adversarial_model.predict(test_adv_x, verbose=0), axis=1))
        
        print(f"Standard model adversarial accuracy: {standard_adv_acc:.4f}")
        print(f"Adversarial model adversarial accuracy: {adversarial_adv_acc:.4f}")
        
        return {
            'standard_model': standard_model,
            'adversarial_model': adversarial_model,
            'clean_acc_standard': standard_acc,
            'clean_acc_adversarial': adversarial_acc,
            'adv_acc_standard': standard_adv_acc,
            'adv_acc_adversarial': adversarial_adv_acc
        }
    
    def implement_defensive_distillation(self, X_train, y_train, X_test, y_test, 
                                       input_shape, num_classes, temperature=10):
        """Implement defensive distillation technique"""
        print("=== DEFENSIVE DISTILLATION ===")
        
        # Step 1: Train teacher model at high temperature
        print("Training teacher model with high temperature...")
        teacher_model = self.create_neural_network_model(input_shape, num_classes)
        
        # Custom loss function with temperature
        def distillation_loss(y_true, y_pred):
            return keras.losses.sparse_categorical_crossentropy(y_true, y_pred / temperature)
        
        teacher_model.compile(
            optimizer='adam',
            loss=distillation_loss,
            metrics=['accuracy']
        )
        
        teacher_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.2)
        
        # Step 2: Generate soft targets from teacher
        print("Generating soft targets from teacher model...")
        soft_targets = teacher_model.predict(X_train, verbose=0)
        soft_targets = tf.nn.softmax(soft_targets / temperature).numpy()
        
        # Step 3: Train student model with soft targets
        print("Training student model with soft targets...")
        student_model = self.create_neural_network_model(input_shape, num_classes)
        
        def soft_target_loss(y_true, y_pred):
            return keras.losses.categorical_crossentropy(y_true, y_pred / temperature)
        
        student_model.compile(
            optimizer='adam',
            loss=soft_target_loss,
            metrics=['accuracy']
        )
        
        # Convert soft targets to proper format
        student_model.fit(X_train, soft_targets, epochs=10, batch_size=32, verbose=0, validation_split=0.2)
        
        # Evaluate robustness
        print("Evaluating defensive distillation...")
        
        # Standard evaluation
        normal_model = self.create_neural_network_model(input_shape, num_classes)
        normal_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.2)
        
        normal_acc = normal_model.evaluate(X_test, y_test, verbose=0)[1]
        distilled_acc = accuracy_score(y_test, np.argmax(student_model.predict(X_test, verbose=0), axis=1))
        
        print(f"Normal model accuracy: {normal_acc:.4f}")
        print(f"Distilled model accuracy: {distilled_acc:.4f}")
        
        # Adversarial evaluation
        test_adv_x = self.generate_fgsm_attack(normal_model, X_test[:100], y_test[:100], epsilon=0.1)
        
        normal_adv_acc = accuracy_score(y_test[:100], 
                                      np.argmax(normal_model.predict(test_adv_x, verbose=0), axis=1))
        distilled_adv_acc = accuracy_score(y_test[:100], 
                                         np.argmax(student_model.predict(test_adv_x, verbose=0), axis=1))
        
        print(f"Normal model adversarial accuracy: {normal_adv_acc:.4f}")
        print(f"Distilled model adversarial accuracy: {distilled_adv_acc:.4f}")
        
        return {
            'normal_model': normal_model,
            'distilled_model': student_model,
            'clean_acc_normal': normal_acc,
            'clean_acc_distilled': distilled_acc,
            'adv_acc_normal': normal_adv_acc,
            'adv_acc_distilled': distilled_adv_acc
        }
    
    def implement_input_preprocessing_defense(self, X_train, y_train, X_test, y_test, 
                                            input_shape, num_classes):
        """Implement input preprocessing defenses"""
        print("=== INPUT PREPROCESSING DEFENSES ===")
        
        def add_gaussian_noise(x, std=0.1):
            """Add Gaussian noise to inputs"""
            noise = np.random.normal(0, std, x.shape)
            return np.clip(x + noise, 0, 1)
        
        def median_filter_defense(x, filter_size=3):
            """Apply median filtering (simplified 1D version)"""
            filtered_x = np.copy(x)
            for i in range(x.shape[1]):
                # Simple median filtering for demonstration
                if i >= filter_size // 2 and i < x.shape[1] - filter_size // 2:
                    window = x[:, i - filter_size // 2:i + filter_size // 2 + 1]
                    filtered_x[:, i] = np.median(window, axis=1)
            return filtered_x
        
        def bit_depth_reduction(x, bits=4):
            """Reduce bit depth of inputs"""
            factor = 2 ** bits
            return np.round(x * factor) / factor
        
        # Train base model
        base_model = self.create_neural_network_model(input_shape, num_classes)
        base_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.2)
        
        # Generate adversarial examples
        adv_test_x = self.generate_fgsm_attack(base_model, X_test[:100], y_test[:100], epsilon=0.1)
        
        # Test different preprocessing defenses
        defenses = {
            'none': lambda x: x,
            'gaussian_noise': lambda x: add_gaussian_noise(x, 0.05),
            'median_filter': median_filter_defense,
            'bit_reduction': lambda x: bit_depth_reduction(x, 6)
        }
        
        results = {}
        
        for defense_name, defense_func in defenses.items():
            print(f"\\nTesting {defense_name} defense...")
            
            # Apply defense to adversarial examples
            defended_adv_x = defense_func(adv_test_x)
            defended_clean_x = defense_func(X_test[:100])
            
            # Evaluate
            clean_acc = accuracy_score(y_test[:100], 
                                     np.argmax(base_model.predict(defended_clean_x, verbose=0), axis=1))
            adv_acc = accuracy_score(y_test[:100], 
                                   np.argmax(base_model.predict(defended_adv_x, verbose=0), axis=1))
            
            results[defense_name] = {
                'clean_accuracy': clean_acc,
                'adversarial_accuracy': adv_acc
            }
            
            print(f"  Clean accuracy: {clean_acc:.4f}")
            print(f"  Adversarial accuracy: {adv_acc:.4f}")
        
        return results, base_model
    
    def implement_adversarial_detection(self, model, X_clean, X_adversarial, feature_extractor=None):
        """Implement adversarial example detection"""
        print("=== ADVERSARIAL DETECTION ===")
        
        def extract_detection_features(x, model):
            """Extract features for adversarial detection"""
            # Get model predictions and confidence
            predictions = model.predict(x, verbose=0)
            confidence = np.max(predictions, axis=1)
            entropy = -np.sum(predictions * np.log(predictions + 1e-8), axis=1)
            
            # Statistical features of input
            input_mean = np.mean(x, axis=1)
            input_std = np.std(x, axis=1)
            input_max = np.max(x, axis=1)
            input_min = np.min(x, axis=1)
            
            features = np.column_stack([
                confidence, entropy, input_mean, input_std, input_max, input_min
            ])
            
            return features
        
        # Extract features
        clean_features = extract_detection_features(X_clean, model)
        adv_features = extract_detection_features(X_adversarial, model)
        
        # Create detection dataset
        X_detection = np.vstack([clean_features, adv_features])
        y_detection = np.hstack([np.zeros(len(clean_features)), np.ones(len(adv_features))])
        
        # Split detection data
        X_det_train, X_det_test, y_det_train, y_det_test = train_test_split(
            X_detection, y_detection, test_size=0.3, random_state=42, stratify=y_detection
        )
        
        # Train detector
        detector = RandomForestClassifier(n_estimators=100, random_state=42)
        detector.fit(X_det_train, y_det_train)
        
        # Evaluate detector
        det_accuracy = detector.score(X_det_test, y_det_test)
        det_predictions = detector.predict(X_det_test)
        
        # Calculate detection rates
        clean_indices = y_det_test == 0
        adv_indices = y_det_test == 1
        
        clean_detection_rate = np.mean(det_predictions[clean_indices] == 0)  # True negative rate
        adv_detection_rate = np.mean(det_predictions[adv_indices] == 1)      # True positive rate
        
        print(f"Detector accuracy: {det_accuracy:.4f}")
        print(f"Clean sample correct classification rate: {clean_detection_rate:.4f}")
        print(f"Adversarial sample detection rate: {adv_detection_rate:.4f}")
        
        return {
            'detector': detector,
            'accuracy': det_accuracy,
            'clean_rate': clean_detection_rate,
            'detection_rate': adv_detection_rate,
            'feature_importance': detector.feature_importances_
        }
    
    def evaluate_robustness_metrics(self, model, X_test, y_test, epsilons=[0.01, 0.05, 0.1, 0.15, 0.2]):
        """Evaluate model robustness across different perturbation magnitudes"""
        print("=== ROBUSTNESS EVALUATION ===")
        
        robustness_results = []
        
        for epsilon in epsilons:
            print(f"Testing epsilon = {epsilon}")
            
            # Generate adversarial examples
            adv_x = self.generate_fgsm_attack(model, X_test[:200], y_test[:200], epsilon)
            
            # Evaluate
            clean_acc = accuracy_score(y_test[:200], 
                                     np.argmax(model.predict(X_test[:200], verbose=0), axis=1))
            adv_acc = accuracy_score(y_test[:200], 
                                   np.argmax(model.predict(adv_x, verbose=0), axis=1))
            
            # Calculate attack success rate
            attack_success_rate = 1 - adv_acc
            
            robustness_results.append({
                'epsilon': epsilon,
                'clean_accuracy': clean_acc,
                'adversarial_accuracy': adv_acc,
                'attack_success_rate': attack_success_rate,
                'robustness_score': min(clean_acc, adv_acc)  # Conservative robustness measure
            })
            
            print(f"  Clean accuracy: {clean_acc:.4f}")
            print(f"  Adversarial accuracy: {adv_acc:.4f}")
            print(f"  Attack success rate: {attack_success_rate:.4f}")
        
        return robustness_results
    
    def visualize_defense_analysis(self, adversarial_training_results, distillation_results, 
                                 preprocessing_results, detection_results, robustness_results):
        """Visualize comprehensive defense analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Adversarial Training Comparison
        methods = ['Standard', 'Adversarial']
        clean_accs = [adversarial_training_results['clean_acc_standard'], 
                     adversarial_training_results['clean_acc_adversarial']]
        adv_accs = [adversarial_training_results['adv_acc_standard'], 
                   adversarial_training_results['adv_acc_adversarial']]
        
        x = np.arange(len(methods))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, clean_accs, width, label='Clean Accuracy', alpha=0.8)
        axes[0, 0].bar(x + width/2, adv_accs, width, label='Adversarial Accuracy', alpha=0.8)
        axes[0, 0].set_xlabel('Training Method')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Adversarial Training Results')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(methods)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1)
        
        # 2. Defensive Distillation Comparison
        methods = ['Normal', 'Distilled']
        clean_accs = [distillation_results['clean_acc_normal'], 
                     distillation_results['clean_acc_distilled']]
        adv_accs = [distillation_results['adv_acc_normal'], 
                   distillation_results['adv_acc_distilled']]
        
        axes[0, 1].bar(x - width/2, clean_accs, width, label='Clean Accuracy', alpha=0.8)
        axes[0, 1].bar(x + width/2, adv_accs, width, label='Adversarial Accuracy', alpha=0.8)
        axes[0, 1].set_xlabel('Model Type')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Defensive Distillation Results')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(methods)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)
        
        # 3. Preprocessing Defenses
        defense_names = list(preprocessing_results.keys())
        clean_accs = [preprocessing_results[name]['clean_accuracy'] for name in defense_names]
        adv_accs = [preprocessing_results[name]['adversarial_accuracy'] for name in defense_names]
        
        x = np.arange(len(defense_names))
        axes[0, 2].bar(x - width/2, clean_accs, width, label='Clean Accuracy', alpha=0.8)
        axes[0, 2].bar(x + width/2, adv_accs, width, label='Adversarial Accuracy', alpha=0.8)
        axes[0, 2].set_xlabel('Defense Method')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].set_title('Input Preprocessing Defenses')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(defense_names, rotation=45)
        axes[0, 2].legend()
        axes[0, 2].set_ylim(0, 1)
        
        # 4. Detection Performance
        detection_metrics = ['Accuracy', 'Clean Rate', 'Detection Rate']
        detection_scores = [detection_results['accuracy'], 
                          detection_results['clean_rate'], 
                          detection_results['detection_rate']]
        
        axes[1, 0].bar(detection_metrics, detection_scores, alpha=0.8, color='orange')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Adversarial Detection Performance')
        axes[1, 0].set_ylim(0, 1)
        
        # 5. Robustness vs Epsilon
        epsilons = [r['epsilon'] for r in robustness_results]
        clean_accs = [r['clean_accuracy'] for r in robustness_results]
        adv_accs = [r['adversarial_accuracy'] for r in robustness_results]
        
        axes[1, 1].plot(epsilons, clean_accs, 'o-', label='Clean Accuracy', linewidth=2)
        axes[1, 1].plot(epsilons, adv_accs, 's-', label='Adversarial Accuracy', linewidth=2)
        axes[1, 1].set_xlabel('Perturbation Magnitude (ε)')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Robustness vs Perturbation Magnitude')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Defense Feature Importance
        feature_names = ['Confidence', 'Entropy', 'Mean', 'Std', 'Max', 'Min']
        importance = detection_results['feature_importance']
        
        axes[1, 2].bar(feature_names, importance, alpha=0.8, color='green')
        axes[1, 2].set_xlabel('Feature')
        axes[1, 2].set_ylabel('Importance')
        axes[1, 2].set_title('Detection Feature Importance')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

# Demonstrate adversarial defense framework
defense_framework = AdversarialDefenseFramework()

print("=== ADVERSARIAL ATTACK DEFENSE FRAMEWORK ===")

# Prepare data
digits = load_digits()
X, y = digits.data, digits.target

# Normalize to [0, 1]
X = X / 16.0

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

input_shape = X.shape[1]
num_classes = len(np.unique(y))

print(f"Dataset: {len(X_train)} training samples, {len(X_test)} test samples")
print(f"Input shape: {input_shape}, Classes: {num_classes}")

# Run comprehensive defense analysis

# 1. Adversarial Training
print("\n" + "="*50)
adversarial_training_results = defense_framework.implement_adversarial_training(
    X_train, y_train, X_test, y_test, input_shape, num_classes
)

# 2. Defensive Distillation
print("\n" + "="*50)
distillation_results = defense_framework.implement_defensive_distillation(
    X_train, y_train, X_test, y_test, input_shape, num_classes
)

# 3. Input Preprocessing Defenses
print("\n" + "="*50)
preprocessing_results, base_model = defense_framework.implement_input_preprocessing_defense(
    X_train, y_train, X_test, y_test, input_shape, num_classes
)

# 4. Adversarial Detection
print("\n" + "="*50)
clean_samples = X_test[:100]
adv_samples = defense_framework.generate_fgsm_attack(base_model, X_test[:100], y_test[:100])

detection_results = defense_framework.implement_adversarial_detection(
    base_model, clean_samples, adv_samples
)

# 5. Robustness Evaluation
print("\n" + "="*50)
robustness_results = defense_framework.evaluate_robustness_metrics(base_model, X_test, y_test)

# Visualize comprehensive results
defense_framework.visualize_defense_analysis(
    adversarial_training_results, distillation_results, preprocessing_results,
    detection_results, robustness_results
)
```

## Summary

This chapter provided comprehensive techniques for protecting AI systems from adversarial attacks:

### Key Defense Mechanisms:
1. **Adversarial Training**: Training with adversarial examples to improve robustness
2. **Defensive Distillation**: Using soft targets to reduce gradient information
3. **Input Preprocessing**: Noise addition, filtering, and transformations
4. **Adversarial Detection**: Identifying adversarial inputs before processing
5. **Certified Defenses**: Provable robustness guarantees

### Defense Strategies:
- **Proactive Defenses**: Built into the model training process
- **Reactive Defenses**: Applied during inference time
- **Detection-Based**: Identify and reject adversarial inputs
- **Transformation-Based**: Preprocess inputs to remove adversarial perturbations
- **Ensemble Methods**: Combine multiple defense mechanisms

### Best Practices:
- Use multiple defense layers (defense in depth)
- Regularly evaluate against new attack methods
- Balance clean accuracy with adversarial robustness
- Consider computational overhead of defense mechanisms
- Implement continuous monitoring and updating of defenses

---

## Exercises

1. **Custom Defense**: Implement a novel defense mechanism
2. **Attack Evaluation**: Test defenses against multiple attack types
3. **Adaptive Attacks**: Design attacks that adapt to specific defenses
4. **Certified Robustness**: Implement provable defense guarantees
5. **Real-world Deployment**: Build production-ready defense system

---

*Effective adversarial defense requires understanding attack mechanisms and implementing layered defense strategies that balance robustness with performance.* 