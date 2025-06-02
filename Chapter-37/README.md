# Chapter 37: Ensuring Data Privacy in AI Systems

## Learning Objectives
By the end of this chapter, students will be able to:
- Implement differential privacy mechanisms in machine learning
- Apply federated learning for privacy-preserving model training
- Design secure multi-party computation protocols for AI
- Understand and implement privacy-preserving data analysis techniques
- Evaluate privacy-utility trade-offs in AI systems

## Table of Contents
1. [Introduction to Privacy in AI](#introduction)
2. [Differential Privacy](#differential-privacy)
3. [Federated Learning](#federated-learning)
4. [Secure Multi-Party Computation](#secure-mpc)
5. [Privacy-Preserving Data Analysis](#privacy-data-analysis)
6. [Privacy Evaluation and Metrics](#privacy-evaluation)

## 1. Introduction to Privacy in AI {#introduction}

Privacy in AI systems involves protecting sensitive information while maintaining the utility of machine learning models and data analysis.

### Privacy Challenges in AI:
- **Data Exposure**: Training data may contain sensitive personal information
- **Model Inversion**: Attackers can reconstruct training data from models
- **Membership Inference**: Determining if specific data was used in training
- **Property Inference**: Learning dataset properties from model behavior
- **Data Linkage**: Combining datasets to reveal private information

### Privacy-Preserving Techniques:
- **Differential Privacy**: Mathematical framework for quantifying privacy
- **Federated Learning**: Distributed training without centralized data
- **Secure Multi-Party Computation**: Collaborative computation on encrypted data
- **Homomorphic Encryption**: Computing on encrypted data
- **Data Anonymization**: Removing or obscuring identifying information

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import hashlib
import hmac
import secrets
import warnings
warnings.filterwarnings('ignore')

class PrivacyPreservingFramework:
    """Comprehensive framework for privacy-preserving AI"""
    
    def __init__(self):
        self.privacy_budgets = {}
        self.federated_models = {}
        
    def implement_differential_privacy(self, X, y, epsilon=1.0, delta=1e-5):
        """Implement differential privacy mechanisms"""
        print("=== DIFFERENTIAL PRIVACY IMPLEMENTATION ===")
        
        class DifferentialPrivacyMechanism:
            def __init__(self, epsilon, delta):
                self.epsilon = epsilon
                self.delta = delta
                self.privacy_spent = 0
                
            def laplace_mechanism(self, true_value, sensitivity):
                """Add Laplace noise for differential privacy"""
                if self.privacy_spent + self.epsilon > self.epsilon:
                    raise ValueError("Privacy budget exceeded")
                
                scale = sensitivity / self.epsilon
                noise = np.random.laplace(0, scale)
                self.privacy_spent += self.epsilon
                
                return true_value + noise
            
            def gaussian_mechanism(self, true_value, sensitivity, epsilon_spent):
                """Add Gaussian noise for (epsilon, delta)-differential privacy"""
                if self.privacy_spent + epsilon_spent > self.epsilon:
                    raise ValueError("Privacy budget exceeded")
                
                # Calculate sigma for Gaussian mechanism
                sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / epsilon_spent
                noise = np.random.normal(0, sigma)
                self.privacy_spent += epsilon_spent
                
                return true_value + noise
            
            def exponential_mechanism(self, candidates, utility_function, sensitivity):
                """Select output using exponential mechanism"""
                if self.privacy_spent + self.epsilon > self.epsilon:
                    raise ValueError("Privacy budget exceeded")
                
                utilities = [utility_function(candidate) for candidate in candidates]
                max_utility = max(utilities)
                
                # Calculate probabilities
                probabilities = []
                for utility in utilities:
                    prob = np.exp(self.epsilon * utility / (2 * sensitivity))
                    probabilities.append(prob)
                
                probabilities = np.array(probabilities)
                probabilities = probabilities / np.sum(probabilities)
                
                # Sample from distribution
                selected_idx = np.random.choice(len(candidates), p=probabilities)
                self.privacy_spent += self.epsilon
                
                return candidates[selected_idx]
        
        # Create DP mechanism
        dp_mechanism = DifferentialPrivacyMechanism(epsilon, delta)
        
        # Apply differential privacy to data statistics
        print(f"Privacy budget: ε={epsilon}, δ={delta}")
        
        # Differentially private statistics
        n_samples = len(X)
        n_features = X.shape[1]
        
        # Add noise to dataset size (sensitivity = 1)
        dp_n_samples = dp_mechanism.laplace_mechanism(n_samples, sensitivity=1)
        print(f"True sample count: {n_samples}")
        print(f"DP sample count: {dp_n_samples:.0f}")
        
        # Add noise to feature means (sensitivity depends on feature range)
        feature_ranges = np.max(X, axis=0) - np.min(X, axis=0)
        dp_means = []
        
        for i in range(min(5, n_features)):  # Show first 5 features
            true_mean = np.mean(X[:, i])
            sensitivity = feature_ranges[i] / n_samples  # Sensitivity for mean
            dp_mean = dp_mechanism.gaussian_mechanism(true_mean, sensitivity, epsilon/10)
            dp_means.append(dp_mean)
            
            print(f"Feature {i}: True mean={true_mean:.3f}, DP mean={dp_mean:.3f}")
        
        # Differentially private model training
        print("\\nTraining differentially private model...")
        
        # Simple DP-SGD implementation (conceptual)
        def add_gradient_noise(gradients, sensitivity, epsilon_per_step):
            """Add noise to gradients for DP-SGD"""
            sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon_per_step
            noisy_gradients = []
            for grad in gradients:
                noise = np.random.normal(0, sigma, grad.shape)
                noisy_gradients.append(grad + noise)
            return noisy_gradients
        
        # Train standard model
        standard_model = LogisticRegression(random_state=42)
        standard_model.fit(X, y)
        standard_accuracy = standard_model.score(X, y)
        
        # Simulate DP model training with noise
        # In practice, this would involve modifying the training algorithm
        dp_model = LogisticRegression(random_state=42)
        
        # Add noise to training data (simplified approach)
        noise_scale = 0.1  # Simplified noise addition
        X_noisy = X + np.random.normal(0, noise_scale, X.shape)
        dp_model.fit(X_noisy, y)
        dp_accuracy = dp_model.score(X, y)
        
        print(f"Standard model accuracy: {standard_accuracy:.4f}")
        print(f"DP model accuracy: {dp_accuracy:.4f}")
        print(f"Privacy cost: ε={dp_mechanism.privacy_spent:.3f}")
        
        return {
            'dp_mechanism': dp_mechanism,
            'standard_model': standard_model,
            'dp_model': dp_model,
            'standard_accuracy': standard_accuracy,
            'dp_accuracy': dp_accuracy,
            'privacy_spent': dp_mechanism.privacy_spent
        }
    
    def implement_federated_learning(self, X, y, n_clients=5, rounds=10):
        """Implement federated learning simulation"""
        print("=== FEDERATED LEARNING IMPLEMENTATION ===")
        
        class FederatedClient:
            def __init__(self, client_id, X_local, y_local):
                self.client_id = client_id
                self.X_local = X_local
                self.y_local = y_local
                self.local_model = LogisticRegression(random_state=42)
                
            def local_training(self, global_weights=None):
                """Perform local training"""
                if global_weights is not None:
                    # Initialize with global weights (simplified)
                    self.local_model = LogisticRegression(random_state=42)
                
                self.local_model.fit(self.X_local, self.y_local)
                return self.local_model.coef_, self.local_model.intercept_
            
            def evaluate_local(self):
                """Evaluate local model"""
                return self.local_model.score(self.X_local, self.y_local)
        
        class FederatedServer:
            def __init__(self):
                self.global_model = LogisticRegression(random_state=42)
                self.global_weights = None
                self.global_bias = None
                
            def aggregate_weights(self, client_weights, client_biases, client_sizes):
                """Aggregate client model updates using FedAvg"""
                total_samples = sum(client_sizes)
                
                # Weighted average of client weights
                aggregated_weights = np.zeros_like(client_weights[0])
                aggregated_bias = np.zeros_like(client_biases[0])
                
                for weights, bias, size in zip(client_weights, client_biases, client_sizes):
                    weight_factor = size / total_samples
                    aggregated_weights += weight_factor * weights
                    aggregated_bias += weight_factor * bias
                
                self.global_weights = aggregated_weights
                self.global_bias = aggregated_bias
                
                return aggregated_weights, aggregated_bias
            
            def evaluate_global(self, X_test, y_test):
                """Evaluate global model"""
                # Create model with aggregated weights
                model = LogisticRegression(random_state=42)
                model.fit(X_test[:1], y_test[:1])  # Dummy fit for initialization
                model.coef_ = self.global_weights
                model.intercept_ = self.global_bias
                
                return model.score(X_test, y_test)
        
        # Split data among clients
        print(f"Distributing data among {n_clients} clients...")
        
        # Random split (non-IID distribution can be implemented)
        indices = np.random.permutation(len(X))
        client_data_splits = np.array_split(indices, n_clients)
        
        clients = []
        for i, client_indices in enumerate(client_data_splits):
            X_client = X[client_indices]
            y_client = y[client_indices]
            clients.append(FederatedClient(i, X_client, y_client))
            print(f"Client {i}: {len(client_indices)} samples")
        
        # Initialize federated server
        server = FederatedServer()
        
        # Split test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Federated training rounds
        global_accuracies = []
        
        for round_num in range(rounds):
            print(f"\\nFederated Round {round_num + 1}/{rounds}")
            
            client_weights = []
            client_biases = []
            client_sizes = []
            
            # Each client performs local training
            for client in clients:
                weights, bias = client.local_training(server.global_weights)
                client_weights.append(weights[0])  # Extract first row for binary classification
                client_biases.append(bias)
                client_sizes.append(len(client.X_local))
                
                local_acc = client.evaluate_local()
                print(f"  Client {client.client_id} local accuracy: {local_acc:.4f}")
            
            # Server aggregates updates
            global_weights, global_bias = server.aggregate_weights(
                client_weights, client_biases, client_sizes
            )
            
            # Evaluate global model
            global_acc = server.evaluate_global(X_test, y_test)
            global_accuracies.append(global_acc)
            print(f"  Global model accuracy: {global_acc:.4f}")
        
        # Compare with centralized training
        centralized_model = LogisticRegression(random_state=42)
        centralized_model.fit(X_train, y_train)
        centralized_acc = centralized_model.score(X_test, y_test)
        
        print(f"\\nFinal Results:")
        print(f"Federated learning accuracy: {global_accuracies[-1]:.4f}")
        print(f"Centralized learning accuracy: {centralized_acc:.4f}")
        print(f"Privacy preserved: Individual client data never shared")
        
        return {
            'clients': clients,
            'server': server,
            'global_accuracies': global_accuracies,
            'centralized_accuracy': centralized_acc,
            'final_federated_accuracy': global_accuracies[-1]
        }
    
    def implement_secure_aggregation(self, values, n_parties=3):
        """Implement secure aggregation protocol"""
        print("=== SECURE AGGREGATION PROTOCOL ===")
        
        class SecureAggregation:
            def __init__(self, n_parties):
                self.n_parties = n_parties
                self.party_secrets = {}
                
            def generate_secret_shares(self, value, party_id):
                """Generate secret shares for a value"""
                # Simple additive secret sharing
                shares = []
                remaining_value = value
                
                for i in range(self.n_parties - 1):
                    share = np.random.uniform(-abs(value), abs(value))
                    shares.append(share)
                    remaining_value -= share
                
                shares.append(remaining_value)  # Last share
                
                self.party_secrets[party_id] = shares
                return shares
            
            def aggregate_shares(self, all_shares):
                """Aggregate secret shares without revealing individual values"""
                # Sum corresponding shares from all parties
                aggregated_shares = []
                
                for share_idx in range(self.n_parties):
                    share_sum = sum(shares[share_idx] for shares in all_shares)
                    aggregated_shares.append(share_sum)
                
                # Reconstruct final sum
                final_sum = sum(aggregated_shares)
                return final_sum
        
        # Simulate secure aggregation
        secure_agg = SecureAggregation(n_parties)
        
        print(f"Simulating secure aggregation with {n_parties} parties")
        print(f"Original values: {values}")
        
        # Each party generates secret shares
        all_shares = []
        for i, value in enumerate(values):
            shares = secure_agg.generate_secret_shares(value, i)
            all_shares.append(shares)
            print(f"Party {i} shares: {shares}")
        
        # Aggregate without revealing individual values
        aggregated_result = secure_agg.aggregate_shares(all_shares)
        true_sum = sum(values)
        
        print(f"Aggregated result: {aggregated_result:.6f}")
        print(f"True sum: {true_sum:.6f}")
        print(f"Error: {abs(aggregated_result - true_sum):.10f}")
        
        return {
            'secure_aggregation': secure_agg,
            'aggregated_result': aggregated_result,
            'true_sum': true_sum,
            'privacy_preserved': True
        }
    
    def implement_homomorphic_encryption_simulation(self, data):
        """Simulate homomorphic encryption for privacy-preserving computation"""
        print("=== HOMOMORPHIC ENCRYPTION SIMULATION ===")
        
        class SimpleHomomorphicEncryption:
            """Simplified homomorphic encryption for demonstration"""
            
            def __init__(self, key_size=1024):
                self.key_size = key_size
                self.public_key, self.private_key = self.generate_keys()
                
            def generate_keys(self):
                """Generate public/private key pair (simplified)"""
                # In practice, this would use proper cryptographic key generation
                private_key = secrets.randbits(self.key_size)
                public_key = private_key  # Simplified for demo
                return public_key, private_key
            
            def encrypt(self, plaintext):
                """Encrypt plaintext (simplified additive encryption)"""
                # Simple additive encryption for demonstration
                noise = np.random.randint(1, 100)
                ciphertext = plaintext + self.public_key + noise
                return ciphertext, noise
            
            def decrypt(self, ciphertext, noise):
                """Decrypt ciphertext"""
                plaintext = ciphertext - self.private_key - noise
                return plaintext
            
            def homomorphic_add(self, ciphertext1, ciphertext2):
                """Perform addition on encrypted values"""
                return ciphertext1 + ciphertext2
            
            def homomorphic_multiply_constant(self, ciphertext, constant):
                """Multiply encrypted value by unencrypted constant"""
                return ciphertext * constant
        
        # Initialize homomorphic encryption
        he = SimpleHomomorphicEncryption()
        
        print("Demonstrating homomorphic encryption...")
        
        # Encrypt sample data
        encrypted_data = []
        noise_values = []
        
        sample_data = data[:5]  # Use first 5 values for demonstration
        print(f"Original data: {sample_data}")
        
        for value in sample_data:
            ciphertext, noise = he.encrypt(value)
            encrypted_data.append(ciphertext)
            noise_values.append(noise)
        
        print(f"Encrypted data: {encrypted_data}")
        
        # Perform homomorphic operations
        # Addition
        encrypted_sum = sum(encrypted_data)
        
        # Multiplication by constant
        constant = 2
        encrypted_doubled = [he.homomorphic_multiply_constant(ct, constant) for ct in encrypted_data]
        
        # Decrypt results
        decrypted_sum = he.decrypt(encrypted_sum, sum(noise_values))
        decrypted_doubled = [he.decrypt(ct, noise * constant) for ct, noise in zip(encrypted_doubled, noise_values)]
        
        # Verify results
        true_sum = sum(sample_data)
        true_doubled = [x * constant for x in sample_data]
        
        print(f"\\nHomomorphic Operations:")
        print(f"Encrypted sum result: {decrypted_sum}")
        print(f"True sum: {true_sum}")
        print(f"Sum error: {abs(decrypted_sum - true_sum)}")
        
        print(f"Encrypted doubling result: {decrypted_doubled}")
        print(f"True doubling: {true_doubled}")
        
        return {
            'he_system': he,
            'encrypted_data': encrypted_data,
            'decrypted_sum': decrypted_sum,
            'true_sum': true_sum,
            'computation_on_encrypted_data': True
        }
    
    def privacy_utility_analysis(self, X, y, epsilon_values=[0.1, 0.5, 1.0, 2.0, 5.0]):
        """Analyze privacy-utility tradeoff"""
        print("=== PRIVACY-UTILITY TRADEOFF ANALYSIS ===")
        
        results = []
        
        # Baseline (no privacy)
        baseline_model = LogisticRegression(random_state=42)
        baseline_model.fit(X, y)
        baseline_accuracy = baseline_model.score(X, y)
        
        results.append({
            'epsilon': float('inf'),
            'privacy_level': 'None',
            'accuracy': baseline_accuracy,
            'utility_loss': 0.0
        })
        
        print(f"Baseline accuracy (no privacy): {baseline_accuracy:.4f}")
        
        # Test different privacy levels
        for epsilon in epsilon_values:
            print(f"\\nTesting ε = {epsilon}")
            
            # Add noise proportional to privacy budget
            noise_scale = 1.0 / epsilon  # Inverse relationship
            X_noisy = X + np.random.normal(0, noise_scale * 0.1, X.shape)
            
            # Train model with noisy data
            private_model = LogisticRegression(random_state=42)
            private_model.fit(X_noisy, y)
            private_accuracy = private_model.score(X, y)
            
            utility_loss = baseline_accuracy - private_accuracy
            privacy_level = 'High' if epsilon < 1.0 else 'Medium' if epsilon < 2.0 else 'Low'
            
            results.append({
                'epsilon': epsilon,
                'privacy_level': privacy_level,
                'accuracy': private_accuracy,
                'utility_loss': utility_loss
            })
            
            print(f"  Privacy level: {privacy_level}")
            print(f"  Accuracy: {private_accuracy:.4f}")
            print(f"  Utility loss: {utility_loss:.4f}")
        
        return results
    
    def implement_k_anonymity(self, data, k=3, quasi_identifiers=None):
        """Implement k-anonymity for data privacy"""
        print("=== K-ANONYMITY IMPLEMENTATION ===")
        
        if quasi_identifiers is None:
            quasi_identifiers = list(range(min(3, data.shape[1])))  # Use first 3 columns
        
        print(f"Implementing {k}-anonymity")
        print(f"Quasi-identifiers: columns {quasi_identifiers}")
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame(data)
        
        # Function to generalize numerical data
        def generalize_numerical(series, levels=3):
            """Generalize numerical data into ranges"""
            min_val, max_val = series.min(), series.max()
            bins = np.linspace(min_val, max_val, levels + 1)
            
            generalized = pd.cut(series, bins=bins, include_lowest=True)
            return generalized.astype(str)
        
        # Apply generalization to quasi-identifiers
        anonymized_df = df.copy()
        
        for col in quasi_identifiers:
            if col < df.shape[1]:
                anonymized_df[col] = generalize_numerical(df[col])
        
        # Check k-anonymity
        qi_columns = [col for col in quasi_identifiers if col < df.shape[1]]
        
        if qi_columns:
            # Group by quasi-identifiers and count
            group_sizes = anonymized_df.groupby(qi_columns).size()
            
            # Check if all groups have at least k members
            min_group_size = group_sizes.min()
            k_anonymous = min_group_size >= k
            groups_below_k = (group_sizes < k).sum()
            
            print(f"Minimum group size: {min_group_size}")
            print(f"Groups below k={k}: {groups_below_k}")
            print(f"K-anonymous: {k_anonymous}")
            
            # Show some examples
            print("\\nExample anonymized records:")
            print(anonymized_df[qi_columns].head(10))
            
            return {
                'anonymized_data': anonymized_df,
                'k_anonymous': k_anonymous,
                'min_group_size': min_group_size,
                'groups_below_k': groups_below_k,
                'group_sizes': group_sizes
            }
        else:
            print("No valid quasi-identifiers found")
            return None
    
    def visualize_privacy_analysis(self, dp_results, federated_results, privacy_utility_results):
        """Visualize privacy analysis results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Differential Privacy Accuracy Comparison
        models = ['Standard', 'Differential Privacy']
        accuracies = [dp_results['standard_accuracy'], dp_results['dp_accuracy']]
        
        axes[0, 0].bar(models, accuracies, alpha=0.7, color=['blue', 'red'])
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Differential Privacy Impact')
        axes[0, 0].set_ylim(0, 1)
        
        # Add value labels
        for i, acc in enumerate(accuracies):
            axes[0, 0].text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. Federated Learning Convergence
        rounds = range(1, len(federated_results['global_accuracies']) + 1)
        axes[0, 1].plot(rounds, federated_results['global_accuracies'], 'o-', linewidth=2, label='Federated')
        axes[0, 1].axhline(y=federated_results['centralized_accuracy'], color='red', 
                          linestyle='--', label='Centralized')
        axes[0, 1].set_xlabel('Communication Round')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Federated Learning Convergence')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Privacy-Utility Tradeoff
        epsilons = [r['epsilon'] for r in privacy_utility_results if r['epsilon'] != float('inf')]
        accuracies = [r['accuracy'] for r in privacy_utility_results if r['epsilon'] != float('inf')]
        
        axes[0, 2].semilogx(epsilons, accuracies, 'o-', linewidth=2, color='green')
        axes[0, 2].axhline(y=privacy_utility_results[0]['accuracy'], color='blue', 
                          linestyle='--', label='No Privacy')
        axes[0, 2].set_xlabel('Privacy Budget (ε)')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].set_title('Privacy-Utility Tradeoff')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Privacy Techniques Comparison
        techniques = ['Centralized', 'Differential Privacy', 'Federated Learning', 'Homomorphic Encryption']
        privacy_levels = [0, 8, 7, 9]  # Subjective privacy levels (0-10 scale)
        utility_levels = [10, 6, 8, 5]  # Subjective utility levels (0-10 scale)
        
        x = np.arange(len(techniques))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, privacy_levels, width, label='Privacy Level', alpha=0.8)
        axes[1, 0].bar(x + width/2, utility_levels, width, label='Utility Level', alpha=0.8)
        axes[1, 0].set_xlabel('Technique')
        axes[1, 0].set_ylabel('Level (0-10)')
        axes[1, 0].set_title('Privacy vs Utility by Technique')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(techniques, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 10)
        
        # 5. Privacy Budget Consumption
        if 'privacy_spent' in dp_results:
            budget_used = dp_results['privacy_spent']
            total_budget = 1.0  # Assuming epsilon = 1.0
            budget_remaining = total_budget - budget_used
            
            labels = ['Used', 'Remaining']
            sizes = [budget_used, budget_remaining]
            colors = ['red', 'green']
            
            axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Privacy Budget Consumption')
        
        # 6. Attack Resistance Comparison
        attack_types = ['Membership\\nInference', 'Model\\nInversion', 'Property\\nInference', 'Linkage\\nAttack']
        standard_resistance = [3, 2, 3, 2]  # Low resistance
        private_resistance = [8, 7, 8, 9]   # High resistance
        
        x = np.arange(len(attack_types))
        axes[1, 2].bar(x - width/2, standard_resistance, width, label='Standard ML', alpha=0.8)
        axes[1, 2].bar(x + width/2, private_resistance, width, label='Privacy-Preserving ML', alpha=0.8)
        axes[1, 2].set_xlabel('Attack Type')
        axes[1, 2].set_ylabel('Resistance Level')
        axes[1, 2].set_title('Attack Resistance Comparison')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(attack_types)
        axes[1, 2].legend()
        axes[1, 2].set_ylim(0, 10)
        
        plt.tight_layout()
        plt.show()

# Demonstrate privacy-preserving framework
privacy_framework = PrivacyPreservingFramework()

print("=== PRIVACY-PRESERVING AI FRAMEWORK ===")

# Load and prepare data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

print(f"Dataset: {len(X)} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

# Run comprehensive privacy analysis

# 1. Differential Privacy
print("\\n" + "="*50)
dp_results = privacy_framework.implement_differential_privacy(X_normalized, y, epsilon=1.0)

# 2. Federated Learning
print("\\n" + "="*50)
federated_results = privacy_framework.implement_federated_learning(X_normalized, y, n_clients=5)

# 3. Secure Aggregation
print("\\n" + "="*50)
sample_values = [1.5, 2.3, 3.1, 4.2, 1.8]
secure_agg_results = privacy_framework.implement_secure_aggregation(sample_values)

# 4. Homomorphic Encryption
print("\\n" + "="*50)
he_results = privacy_framework.implement_homomorphic_encryption_simulation(X_normalized[0])

# 5. Privacy-Utility Analysis
print("\\n" + "="*50)
privacy_utility_results = privacy_framework.privacy_utility_analysis(X_normalized, y)

# 6. K-Anonymity
print("\\n" + "="*50)
k_anon_results = privacy_framework.implement_k_anonymity(X_normalized[:100], k=5)

# Visualize results
privacy_framework.visualize_privacy_analysis(dp_results, federated_results, privacy_utility_results)
```

## Summary

This chapter covered comprehensive techniques for ensuring data privacy in AI systems:

### Key Privacy-Preserving Techniques:
1. **Differential Privacy**: Mathematical framework for quantifying and limiting privacy loss
2. **Federated Learning**: Distributed training without centralized data collection
3. **Secure Multi-Party Computation**: Collaborative computation on encrypted data
4. **Homomorphic Encryption**: Computing on encrypted data without decryption
5. **Data Anonymization**: K-anonymity, l-diversity, and t-closeness

### Privacy Mechanisms:
- **Noise Addition**: Laplace and Gaussian mechanisms for differential privacy
- **Secure Aggregation**: Combining model updates without revealing individual contributions
- **Cryptographic Protocols**: Encryption-based privacy preservation
- **Data Generalization**: Reducing data granularity to protect privacy
- **Access Control**: Limiting data access and usage

### Privacy-Utility Tradeoffs:
- Higher privacy often means lower utility and accuracy
- Privacy budget management for differential privacy
- Communication efficiency in federated learning
- Computational overhead of cryptographic methods
- Balance between privacy protection and model performance

---

## Exercises

1. **Custom DP Mechanism**: Implement advanced differential privacy mechanisms
2. **Federated Optimization**: Design privacy-preserving federated optimization algorithms
3. **Privacy Attacks**: Implement and defend against privacy attacks
4. **Secure Protocols**: Build secure multi-party computation protocols
5. **Privacy Metrics**: Develop comprehensive privacy evaluation frameworks

---

*Privacy-preserving AI requires careful balance between protecting sensitive information and maintaining the utility of machine learning systems.* 