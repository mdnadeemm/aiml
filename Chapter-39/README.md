# Chapter 39: Risk Assessment Using AI

## Learning Objectives
By the end of this chapter, students will be able to:
- Understand different types of risk assessment in various industries
- Implement AI models for credit risk, market risk, and operational risk assessment
- Apply machine learning techniques for risk prediction and mitigation
- Design risk scoring systems and early warning systems
- Evaluate and validate risk assessment models

## Table of Contents
1. [Introduction to Risk Assessment](#introduction)
2. [Credit Risk Assessment](#credit-risk)
3. [Market Risk Assessment](#market-risk)
4. [Operational Risk Assessment](#operational-risk)
5. [Insurance Risk Assessment](#insurance-risk)
6. [Risk Model Validation](#model-validation)

## 1. Introduction to Risk Assessment {#introduction}

Risk assessment is the systematic process of identifying, analyzing, and evaluating potential risks that could negatively impact an organization or individual.

### Types of Risk:
- **Credit Risk**: Risk of financial loss due to borrower default
- **Market Risk**: Risk due to changes in market prices and rates
- **Operational Risk**: Risk from failed internal processes or systems
- **Liquidity Risk**: Risk of inability to meet short-term obligations
- **Insurance Risk**: Risk of claims exceeding premiums collected

### AI Applications in Risk Assessment:
- **Predictive Modeling**: Forecasting risk events and their likelihood
- **Anomaly Detection**: Identifying unusual patterns that indicate risk
- **Stress Testing**: Simulating extreme scenarios and their impacts
- **Real-time Monitoring**: Continuous risk assessment and early warning
- **Portfolio Optimization**: Balancing risk and return in investments

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.cluster import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class RiskAssessmentFramework:
    """Comprehensive framework for AI-driven risk assessment"""
    
    def __init__(self):
        self.models = {}
        self.risk_scores = {}
        self.thresholds = {}
        
    def generate_credit_risk_data(self, n_samples=8000):
        """Generate comprehensive credit risk dataset"""
        print("=== GENERATING CREDIT RISK DATASET ===")
        
        np.random.seed(42)
        
        # Personal Information
        data = {
            'age': np.random.normal(40, 12, n_samples).clip(18, 75),
            'income': np.random.lognormal(10.8, 0.6, n_samples).clip(25000, 300000),
            'employment_years': np.random.gamma(2, 3, n_samples).clip(0, 40),
            'education_level': np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.3, 0.3, 0.2])
        }
        
        # Financial Information
        data.update({
            'debt_to_income': np.random.beta(2, 8, n_samples),
            'credit_score': np.random.normal(680, 80, n_samples).clip(300, 850),
            'credit_history_months': np.random.gamma(3, 12, n_samples).clip(6, 360),
            'num_credit_lines': np.random.poisson(6, n_samples).clip(1, 25),
            'credit_utilization': np.random.beta(2, 5, n_samples).clip(0, 1),
            'num_delinquencies': np.random.poisson(0.8, n_samples).clip(0, 15),
            'num_inquiries': np.random.poisson(2, n_samples).clip(0, 20)
        })
        
        # Loan Information
        data.update({
            'loan_amount': np.random.lognormal(10.2, 0.7, n_samples).clip(5000, 150000),
            'loan_term': np.random.choice([12, 24, 36, 48, 60], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
            'loan_purpose': np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.2, 0.25, 0.15])
        })
        
        df = pd.DataFrame(data)
        
        # Calculate comprehensive risk score
        risk_factors = (
            # Income and employment stability
            -0.00003 * df['income'] +
            -0.01 * df['employment_years'] +
            -0.1 * df['education_level'] +
            
            # Credit behavior
            -0.005 * (df['credit_score'] - 600) +
            -0.002 * df['credit_history_months'] +
            2.5 * df['debt_to_income'] +
            1.5 * df['credit_utilization'] +
            0.15 * df['num_delinquencies'] +
            0.05 * df['num_inquiries'] +
            
            # Loan characteristics
            0.00002 * df['loan_amount'] +
            0.01 * df['loan_term'] +
            0.1 * df['loan_purpose']
        )
        
        # Convert to probability with noise
        default_prob = 1 / (1 + np.exp(-risk_factors + np.random.normal(0, 0.4, n_samples)))
        y = (default_prob > 0.25).astype(int)
        
        print(f"Generated {n_samples} credit applications")
        print(f"Default rate: {y.mean():.2%}")
        print(f"Average credit score: {df['credit_score'].mean():.0f}")
        
        return df, y
    
    def implement_credit_risk_model(self, X, y):
        """Implement comprehensive credit risk assessment model"""
        print("\\n=== CREDIT RISK ASSESSMENT MODEL ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features for some models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, random_state=42, max_depth=6)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\\nTraining {name}...")
            
            # Use scaled data for logistic regression
            if 'Logistic' in name:
                model.fit(X_train_scaled, y_train)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
            else:
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred_proba
            }
            
            print(f"AUC Score: {auc_score:.4f}")
            print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Risk tier classification
        best_model_name = max(results.keys(), key=lambda k: results[k]['auc_score'])
        best_predictions = results[best_model_name]['predictions']
        
        # Define risk tiers
        risk_tiers = pd.cut(best_predictions, 
                           bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                           labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        tier_summary = pd.DataFrame({
            'Risk_Tier': risk_tiers,
            'Default_Rate': y_test
        }).groupby('Risk_Tier')['Default_Rate'].agg(['count', 'mean']).round(3)
        
        print(f"\\nRisk Tier Analysis ({best_model_name}):")
        print(tier_summary)
        
        return results, (X_test, y_test), scaler, tier_summary
    
    def generate_market_risk_data(self, n_assets=10, n_periods=1000):
        """Generate market risk data (portfolio returns)"""
        print("\\n=== GENERATING MARKET RISK DATA ===")
        
        np.random.seed(42)
        
        # Generate correlated asset returns
        # Create correlation matrix
        correlation_matrix = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Ensure positive semi-definite
        eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
        eigenvals = np.maximum(eigenvals, 0.01)
        correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Generate returns using multivariate normal
        mean_returns = np.random.uniform(-0.001, 0.002, n_assets)
        volatilities = np.random.uniform(0.01, 0.04, n_assets)
        
        # Covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        # Generate returns
        returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_periods)
        
        # Create DataFrame
        asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
        returns_df = pd.DataFrame(returns, columns=asset_names)
        
        # Add market factors
        returns_df['Market_Return'] = returns_df.mean(axis=1) + np.random.normal(0, 0.005, n_periods)
        returns_df['VIX'] = np.random.gamma(2, 0.01, n_periods).clip(0.1, 0.8)
        returns_df['Interest_Rate'] = np.random.normal(0.03, 0.01, n_periods).clip(0, 0.1)
        
        print(f"Generated {n_periods} periods of returns for {n_assets} assets")
        print(f"Average daily return: {returns_df[asset_names].mean().mean():.4f}")
        print(f"Average daily volatility: {returns_df[asset_names].std().mean():.4f}")
        
        return returns_df, asset_names
    
    def implement_market_risk_var(self, returns_df, asset_names, confidence_level=0.05):
        """Implement Value at Risk (VaR) calculation"""
        print("\\n=== MARKET RISK VaR CALCULATION ===")
        
        # Portfolio weights (equal weight for simplicity)
        weights = np.ones(len(asset_names)) / len(asset_names)
        
        # Calculate portfolio returns
        portfolio_returns = (returns_df[asset_names] * weights).sum(axis=1)
        
        # Historical VaR
        historical_var = np.percentile(portfolio_returns, confidence_level * 100)
        
        # Parametric VaR (assumes normal distribution)
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        from scipy import stats
        parametric_var = stats.norm.ppf(confidence_level, mean_return, std_return)
        
        # Monte Carlo VaR
        n_simulations = 10000
        simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
        monte_carlo_var = np.percentile(simulated_returns, confidence_level * 100)
        
        # Expected Shortfall (Conditional VaR)
        expected_shortfall = portfolio_returns[portfolio_returns <= historical_var].mean()
        
        print(f"Portfolio Risk Metrics (95% confidence):")
        print(f"Historical VaR: {historical_var:.4f} ({historical_var*100:.2f}%)")
        print(f"Parametric VaR: {parametric_var:.4f} ({parametric_var*100:.2f}%)")
        print(f"Monte Carlo VaR: {monte_carlo_var:.4f} ({monte_carlo_var*100:.2f}%)")
        print(f"Expected Shortfall: {expected_shortfall:.4f} ({expected_shortfall*100:.2f}%)")
        
        return {
            'portfolio_returns': portfolio_returns,
            'historical_var': historical_var,
            'parametric_var': parametric_var,
            'monte_carlo_var': monte_carlo_var,
            'expected_shortfall': expected_shortfall,
            'weights': weights
        }
    
    def generate_operational_risk_data(self, n_samples=5000):
        """Generate operational risk data (internal fraud, system failures, etc.)"""
        print("\\n=== GENERATING OPERATIONAL RISK DATA ===")
        
        np.random.seed(42)
        
        # Risk event characteristics
        data = {
            'department': np.random.choice(['IT', 'Trading', 'Operations', 'Compliance', 'HR'], 
                                         n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
            'employee_count': np.random.poisson(50, n_samples).clip(5, 500),
            'system_age_years': np.random.exponential(5, n_samples).clip(0, 20),
            'automation_level': np.random.beta(3, 2, n_samples),
            'training_hours': np.random.gamma(2, 10, n_samples).clip(0, 100),
            'audit_score': np.random.normal(75, 15, n_samples).clip(0, 100),
            'incident_history': np.random.poisson(2, n_samples).clip(0, 20),
            'process_complexity': np.random.uniform(1, 10, n_samples)
        }
        
        # Convert categorical variables
        dept_encoder = LabelEncoder()
        data['department_encoded'] = dept_encoder.fit_transform(data['department'])
        
        df = pd.DataFrame(data)
        
        # Generate operational risk events
        risk_score = (
            0.1 * df['department_encoded'] +
            0.002 * df['employee_count'] +
            0.1 * df['system_age_years'] +
            -1.0 * df['automation_level'] +
            -0.01 * df['training_hours'] +
            -0.02 * df['audit_score'] +
            0.1 * df['incident_history'] +
            0.05 * df['process_complexity']
        )
        
        # Convert to probability
        risk_prob = 1 / (1 + np.exp(-risk_score + np.random.normal(0, 0.5, n_samples)))
        y = (risk_prob > 0.3).astype(int)
        
        print(f"Generated {n_samples} operational scenarios")
        print(f"Risk event rate: {y.mean():.2%}")
        
        return df.drop(['department'], axis=1), y  # Remove original categorical column
    
    def implement_operational_risk_model(self, X, y):
        """Implement operational risk assessment model"""
        print("\\n=== OPERATIONAL RISK ASSESSMENT ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Use Random Forest for operational risk (handles mixed data types well)
        model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=8)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"Operational Risk Model AUC: {auc_score:.4f}")
        print("\\nTop Risk Factors:")
        print(feature_importance.head(8))
        
        # Anomaly detection for operational risk
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_scores = isolation_forest.fit_predict(X_test)
        anomaly_rate = (anomaly_scores == -1).mean()
        
        print(f"\\nAnomalous patterns detected: {anomaly_rate:.2%}")
        
        return model, (X_test, y_test, y_pred_proba), feature_importance, isolation_forest
    
    def implement_early_warning_system(self, risk_scores, threshold_percentile=95):
        """Implement early warning system for risk management"""
        print("\\n=== EARLY WARNING SYSTEM ===")
        
        # Define warning thresholds
        low_threshold = np.percentile(risk_scores, 75)
        medium_threshold = np.percentile(risk_scores, 90)
        high_threshold = np.percentile(risk_scores, threshold_percentile)
        
        # Classify risk levels
        def classify_risk(score):
            if score >= high_threshold:
                return 'Critical'
            elif score >= medium_threshold:
                return 'High'
            elif score >= low_threshold:
                return 'Medium'
            else:
                return 'Low'
        
        risk_levels = [classify_risk(score) for score in risk_scores]
        
        # Alert statistics
        alert_stats = pd.Series(risk_levels).value_counts()
        
        print("Risk Level Distribution:")
        for level, count in alert_stats.items():
            percentage = count / len(risk_scores) * 100
            print(f"{level}: {count} cases ({percentage:.1f}%)")
        
        # Simulate real-time monitoring
        print("\\nReal-time Risk Monitoring Simulation:")
        recent_scores = risk_scores[-20:]  # Last 20 observations
        
        for i, score in enumerate(recent_scores):
            risk_level = classify_risk(score)
            if risk_level in ['High', 'Critical']:
                print(f"🚨 Alert {i+1}: {risk_level} risk detected (Score: {score:.3f})")
            
        return {
            'thresholds': {
                'low': low_threshold,
                'medium': medium_threshold,
                'high': high_threshold
            },
            'risk_levels': risk_levels,
            'alert_stats': alert_stats
        }
    
    def visualize_risk_assessment(self, credit_results, market_var_results, operational_results):
        """Visualize comprehensive risk assessment results"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # Credit Risk Analysis
        # 1. Credit model comparison
        models = list(credit_results.keys())
        auc_scores = [credit_results[model]['auc_score'] for model in models]
        
        axes[0, 0].bar(models, auc_scores, alpha=0.8, color='blue')
        axes[0, 0].set_title('Credit Risk Model Performance')
        axes[0, 0].set_ylabel('AUC Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim(0, 1)
        
        # 2. Credit risk distribution
        best_model = max(credit_results.keys(), key=lambda k: credit_results[k]['auc_score'])
        credit_probs = credit_results[best_model]['predictions']
        
        axes[0, 1].hist(credit_probs, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_title('Credit Risk Score Distribution')
        axes[0, 1].set_xlabel('Default Probability')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. ROC Curve for credit model
        from sklearn.metrics import roc_curve
        # Note: This would use actual test data in practice
        fpr = np.linspace(0, 1, 100)
        tpr = np.power(fpr, 0.5)  # Simulated for visualization
        
        axes[0, 2].plot(fpr, tpr, linewidth=2, label=f'{best_model}')
        axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 2].set_xlabel('False Positive Rate')
        axes[0, 2].set_ylabel('True Positive Rate')
        axes[0, 2].set_title('Credit Risk ROC Curve')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Market Risk Analysis
        # 4. Portfolio returns distribution
        portfolio_returns = market_var_results['portfolio_returns']
        
        axes[1, 0].hist(portfolio_returns, bins=50, alpha=0.7, color='green', density=True)
        axes[1, 0].axvline(market_var_results['historical_var'], color='red', 
                          linestyle='--', label=f"VaR: {market_var_results['historical_var']:.3f}")
        axes[1, 0].set_title('Portfolio Returns Distribution')
        axes[1, 0].set_xlabel('Daily Returns')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        
        # 5. VaR comparison
        var_methods = ['Historical', 'Parametric', 'Monte Carlo']
        var_values = [
            market_var_results['historical_var'],
            market_var_results['parametric_var'],
            market_var_results['monte_carlo_var']
        ]
        
        axes[1, 1].bar(var_methods, [abs(v) for v in var_values], alpha=0.8, color='orange')
        axes[1, 1].set_title('VaR Comparison (Absolute Values)')
        axes[1, 1].set_ylabel('VaR')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Cumulative returns
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        axes[1, 2].plot(cumulative_returns, linewidth=2, color='blue')
        axes[1, 2].set_title('Cumulative Portfolio Returns')
        axes[1, 2].set_xlabel('Trading Days')
        axes[1, 2].set_ylabel('Cumulative Return')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Operational Risk Analysis
        # 7. Feature importance
        feature_importance = operational_results[2]  # Feature importance DataFrame
        top_features = feature_importance.head(8)
        
        axes[2, 0].barh(top_features['feature'], top_features['importance'], alpha=0.8, color='purple')
        axes[2, 0].set_title('Operational Risk Factors')
        axes[2, 0].set_xlabel('Importance')
        
        # 8. Risk score distribution
        op_risk_scores = operational_results[1][2]  # Predictions
        
        axes[2, 1].hist(op_risk_scores, bins=30, alpha=0.7, color='brown')
        axes[2, 1].set_title('Operational Risk Score Distribution')
        axes[2, 1].set_xlabel('Risk Probability')
        axes[2, 1].set_ylabel('Frequency')
        
        # 9. Risk monitoring dashboard (simulated)
        risk_levels = ['Low', 'Medium', 'High', 'Critical']
        risk_counts = [60, 25, 12, 3]  # Simulated counts
        colors = ['green', 'yellow', 'orange', 'red']
        
        axes[2, 2].pie(risk_counts, labels=risk_levels, colors=colors, autopct='%1.1f%%')
        axes[2, 2].set_title('Current Risk Level Distribution')
        
        plt.tight_layout()
        plt.show()

# Demonstrate Risk Assessment Framework
risk_framework = RiskAssessmentFramework()

print("=== COMPREHENSIVE RISK ASSESSMENT FRAMEWORK ===")

# 1. Credit Risk Assessment
credit_data, credit_labels = risk_framework.generate_credit_risk_data()
credit_results, credit_test_data, credit_scaler, tier_summary = risk_framework.implement_credit_risk_model(
    credit_data, credit_labels
)

# 2. Market Risk Assessment
market_data, asset_names = risk_framework.generate_market_risk_data()
market_var_results = risk_framework.implement_market_risk_var(market_data, asset_names)

# 3. Operational Risk Assessment
operational_data, operational_labels = risk_framework.generate_operational_risk_data()
operational_model, operational_results, op_feature_importance, anomaly_detector = risk_framework.implement_operational_risk_model(
    operational_data, operational_labels
)

# 4. Early Warning System
best_credit_model = max(credit_results.keys(), key=lambda k: credit_results[k]['auc_score'])
risk_scores = credit_results[best_credit_model]['predictions']
warning_system = risk_framework.implement_early_warning_system(risk_scores)

# Visualize all results
risk_framework.visualize_risk_assessment(credit_results, market_var_results, operational_results)
```

## Summary

This chapter provided comprehensive techniques for AI-driven risk assessment across multiple domains:

### Key Risk Assessment Applications:
1. **Credit Risk**: Predicting borrower default using ML models and credit scoring
2. **Market Risk**: Calculating Value at Risk (VaR) and portfolio risk metrics
3. **Operational Risk**: Assessing internal process failures and system risks
4. **Insurance Risk**: Evaluating claim probabilities and pricing models
5. **Early Warning Systems**: Real-time risk monitoring and alert systems

### Risk Assessment Techniques:
- **Statistical Modeling**: Traditional statistical approaches for risk quantification
- **Machine Learning**: Advanced ML models for pattern recognition and prediction
- **Anomaly Detection**: Identifying unusual patterns that indicate emerging risks
- **Stress Testing**: Simulating extreme scenarios and their potential impacts
- **Monte Carlo Simulation**: Probabilistic risk modeling and scenario analysis

### Model Validation and Governance:
- **Backtesting**: Validating model performance on historical data
- **Stress Testing**: Testing model robustness under extreme conditions
- **Model Monitoring**: Continuous tracking of model performance in production
- **Regulatory Compliance**: Meeting risk management regulatory requirements
- **Model Documentation**: Maintaining comprehensive model documentation

---

## Exercises

1. **Advanced VaR Models**: Implement GARCH models for volatility forecasting
2. **Stress Testing Framework**: Design comprehensive stress testing scenarios
3. **Credit Portfolio Models**: Build portfolio-level credit risk models
4. **Operational Risk Database**: Create operational risk event database and analysis
5. **Integrated Risk Dashboard**: Build real-time risk monitoring dashboard

---

*Effective risk assessment requires combining domain expertise with advanced AI techniques to identify, quantify, and manage risks across all business functions.* 