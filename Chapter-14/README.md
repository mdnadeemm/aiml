# Chapter 14: Sentiment Analysis in NLP

## Learning Objectives
By the end of this chapter, students will be able to:
- Understand different approaches to sentiment analysis
- Implement rule-based and machine learning sentiment classifiers
- Apply deep learning techniques for sentiment analysis
- Handle challenges in sentiment analysis (sarcasm, context, etc.)
- Evaluate sentiment analysis models effectively

## Table of Contents
1. [Introduction to Sentiment Analysis](#introduction)
2. [Rule-Based Approaches](#rule-based)
3. [Machine Learning Approaches](#ml-approaches)
4. [Deep Learning for Sentiment Analysis](#deep-learning)
5. [Handling Challenges](#challenges)
6. [Multi-Class and Fine-Grained Sentiment](#multi-class)
7. [Aspect-Based Sentiment Analysis](#aspect-based)
8. [Evaluation and Metrics](#evaluation)

## 1. Introduction to Sentiment Analysis {#introduction}

Sentiment analysis, also known as opinion mining, is the computational study of opinions, sentiments, and emotions expressed in text. It aims to determine the attitude of a speaker or writer with respect to some topic.

### Types of Sentiment Analysis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

class SentimentAnalysisIntro:
    """Introduction to sentiment analysis concepts"""
    
    def __init__(self):
        self.sample_texts = {
            'positive': [
                "I love this product! It's amazing!",
                "Great service, highly recommended!",
                "Best purchase I've ever made.",
                "Fantastic quality and fast delivery.",
                "Absolutely wonderful experience!"
            ],
            'negative': [
                "This product is terrible.",
                "Worst customer service ever!",
                "Complete waste of money.",
                "Poor quality, very disappointed.",
                "Horrible experience, avoid at all costs."
            ],
            'neutral': [
                "The product is okay.",
                "Average service, nothing special.",
                "It's an acceptable option.",
                "Standard quality for the price.",
                "Neither good nor bad."
            ]
        }
    
    def demonstrate_sentiment_types(self):
        """Show different types of sentiment"""
        print("=== TYPES OF SENTIMENT ANALYSIS ===")
        
        # Binary sentiment (positive/negative)
        print("1. BINARY SENTIMENT:")
        for sentiment, texts in self.sample_texts.items():
            if sentiment != 'neutral':
                print(f"  {sentiment.upper()}:")
                for text in texts[:2]:
                    print(f"    - {text}")
        
        # Multi-class sentiment (positive/negative/neutral)
        print("\n2. MULTI-CLASS SENTIMENT:")
        for sentiment, texts in self.sample_texts.items():
            print(f"  {sentiment.upper()}:")
            for text in texts[:2]:
                print(f"    - {text}")
        
        # Fine-grained sentiment (scale 1-5)
        print("\n3. FINE-GRAINED SENTIMENT (1-5 scale):")
        fine_grained_examples = [
            (1, "Absolutely terrible, hate it!"),
            (2, "Not good, disappointed."),
            (3, "It's okay, average product."),
            (4, "Pretty good, satisfied."),
            (5, "Excellent! Love it!")
        ]
        
        for score, text in fine_grained_examples:
            print(f"  {score}/5: {text}")
    
    def sentiment_applications(self):
        """Show real-world applications"""
        print("\n=== SENTIMENT ANALYSIS APPLICATIONS ===")
        
        applications = {
            'Business Intelligence': [
                "Brand monitoring and reputation management",
                "Customer feedback analysis",
                "Product review analysis",
                "Market research and competitor analysis"
            ],
            'Social Media': [
                "Social media monitoring",
                "Political sentiment tracking",
                "Crisis management",
                "Influencer analysis"
            ],
            'Customer Service': [
                "Automated customer support",
                "Priority ticket routing",
                "Customer satisfaction measurement",
                "Proactive issue identification"
            ],
            'Finance': [
                "Stock market sentiment analysis",
                "News impact on trading",
                "Investment decision support",
                "Risk assessment"
            ]
        }
        
        for domain, use_cases in applications.items():
            print(f"\n{domain}:")
            for use_case in use_cases:
                print(f"  • {use_case}")
    
    def quick_sentiment_demo(self):
        """Quick demonstration using TextBlob"""
        print("\n=== QUICK SENTIMENT DEMO (TextBlob) ===")
        
        demo_texts = [
            "I absolutely love this movie!",
            "This movie is terrible.",
            "The movie was okay.",
            "I'm not sure if I like it or not.",
            "Best film ever made!!! 😍",
            "Boring and predictable."
        ]
        
        print(f"{'Text':<35} {'Polarity':<10} {'Subjectivity':<12} {'Sentiment'}")
        print("-" * 70)
        
        for text in demo_texts:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if polarity > 0.1:
                sentiment = "Positive"
            elif polarity < -0.1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            print(f"{text:<35} {polarity:<10.3f} {subjectivity:<12.3f} {sentiment}")

# Demonstrate sentiment analysis introduction
intro = SentimentAnalysisIntro()
intro.demonstrate_sentiment_types()
intro.sentiment_applications()
intro.quick_sentiment_demo()
```

## 2. Rule-Based Approaches {#rule-based}

Rule-based sentiment analysis uses predefined dictionaries and linguistic rules to determine sentiment.

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class RuleBasedSentiment:
    """Rule-based sentiment analysis implementation"""
    
    def __init__(self):
        # Sentiment lexicons (simplified versions)
        self.positive_words = {
            'excellent': 3, 'amazing': 3, 'fantastic': 3, 'wonderful': 3, 'outstanding': 3,
            'great': 2, 'good': 2, 'nice': 2, 'love': 2, 'like': 2, 'happy': 2,
            'okay': 1, 'fine': 1, 'decent': 1, 'acceptable': 1
        }
        
        self.negative_words = {
            'terrible': -3, 'awful': -3, 'horrible': -3, 'disgusting': -3, 'hate': -3,
            'bad': -2, 'poor': -2, 'disappointed': -2, 'sad': -2, 'angry': -2,
            'dislike': -1, 'mediocre': -1, 'boring': -1, 'annoying': -1
        }
        
        # Intensifiers and diminishers
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'absolutely': 2.0, 'completely': 2.0,
            'totally': 1.8, 'really': 1.3, 'quite': 1.2, 'pretty': 1.1
        }
        
        self.diminishers = {
            'slightly': 0.5, 'somewhat': 0.6, 'kind of': 0.7, 'sort of': 0.7,
            'a bit': 0.8, 'a little': 0.8, 'rather': 0.9
        }
        
        # Negation words
        self.negation_words = {
            'not', 'no', 'never', 'nothing', 'nobody', 'none', 'neither',
            'nowhere', 'cannot', "can't", "won't", "shouldn't", "wouldn't",
            "couldn't", "doesn't", "don't", "didn't", "isn't", "aren't", "wasn't", "weren't"
        }
        
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation (except for contractions)
        text = re.sub(r'[^\w\s\']', ' ', text)
        # Tokenize
        tokens = word_tokenize(text)
        return tokens
    
    def simple_lexicon_sentiment(self, text):
        """Simple lexicon-based sentiment analysis"""
        tokens = self.preprocess_text(text)
        
        positive_score = 0
        negative_score = 0
        
        for token in tokens:
            if token in self.positive_words:
                positive_score += self.positive_words[token]
            elif token in self.negative_words:
                negative_score += abs(self.negative_words[token])
        
        total_score = positive_score - negative_score
        
        if total_score > 0:
            sentiment = "Positive"
        elif total_score < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        return {
            'sentiment': sentiment,
            'score': total_score,
            'positive_score': positive_score,
            'negative_score': negative_score
        }
    
    def enhanced_lexicon_sentiment(self, text):
        """Enhanced lexicon with intensifiers and negation"""
        tokens = self.preprocess_text(text)
        
        sentiment_score = 0
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            # Check for intensifiers
            intensifier = 1.0
            if i > 0 and tokens[i-1] in self.intensifiers:
                intensifier = self.intensifiers[tokens[i-1]]
            elif i > 0 and tokens[i-1] in self.diminishers:
                intensifier = self.diminishers[tokens[i-1]]
            
            # Check for negation (within 3 words)
            negation = False
            for j in range(max(0, i-3), i):
                if tokens[j] in self.negation_words:
                    negation = True
                    break
            
            # Calculate sentiment
            if token in self.positive_words:
                score = self.positive_words[token] * intensifier
                if negation:
                    score = -score * 0.8  # Negated positive becomes negative
                sentiment_score += score
            
            elif token in self.negative_words:
                score = self.negative_words[token] * intensifier
                if negation:
                    score = -score * 0.8  # Negated negative becomes positive
                sentiment_score += score
            
            i += 1
        
        # Determine sentiment
        if sentiment_score > 0.5:
            sentiment = "Positive"
        elif sentiment_score < -0.5:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        return {
            'sentiment': sentiment,
            'score': sentiment_score
        }
    
    def analyze_with_rules(self, texts):
        """Analyze multiple texts with rule-based approach"""
        print("=== RULE-BASED SENTIMENT ANALYSIS ===")
        
        results = []
        
        for text in texts:
            simple_result = self.simple_lexicon_sentiment(text)
            enhanced_result = self.enhanced_lexicon_sentiment(text)
            
            results.append({
                'text': text,
                'simple': simple_result,
                'enhanced': enhanced_result
            })
        
        # Display results
        print(f"{'Text':<40} {'Simple':<10} {'Enhanced':<10} {'Simple Score':<12} {'Enhanced Score'}")
        print("-" * 90)
        
        for result in results:
            text = result['text'][:35] + "..." if len(result['text']) > 35 else result['text']
            print(f"{text:<40} {result['simple']['sentiment']:<10} {result['enhanced']['sentiment']:<10} "
                  f"{result['simple']['score']:<12.2f} {result['enhanced']['score']:<12.2f}")
        
        return results
    
    def rule_performance_analysis(self, test_texts_with_labels):
        """Analyze rule-based performance"""
        print("\n=== RULE-BASED PERFORMANCE ANALYSIS ===")
        
        correct_simple = 0
        correct_enhanced = 0
        total = len(test_texts_with_labels)
        
        for text, true_label in test_texts_with_labels:
            simple_result = self.simple_lexicon_sentiment(text)
            enhanced_result = self.enhanced_lexicon_sentiment(text)
            
            if simple_result['sentiment'].lower() == true_label.lower():
                correct_simple += 1
            
            if enhanced_result['sentiment'].lower() == true_label.lower():
                correct_enhanced += 1
        
        simple_accuracy = correct_simple / total
        enhanced_accuracy = correct_enhanced / total
        
        print(f"Simple Lexicon Accuracy: {simple_accuracy:.3f}")
        print(f"Enhanced Lexicon Accuracy: {enhanced_accuracy:.3f}")
        print(f"Improvement: {enhanced_accuracy - simple_accuracy:.3f}")
        
        return simple_accuracy, enhanced_accuracy

# Demonstrate rule-based sentiment analysis
rule_sentiment = RuleBasedSentiment()

# Test texts
test_texts = [
    "I love this product!",
    "This is terrible quality.",
    "Not bad, but could be better.",
    "Absolutely amazing experience!",
    "I don't like this at all.",
    "It's okay, nothing special.",
    "Very disappointed with the service.",
    "Really good value for money."
]

results = rule_sentiment.analyze_with_rules(test_texts)

# Test with labeled data
labeled_test_data = [
    ("I love this movie!", "positive"),
    ("This movie is terrible.", "negative"),
    ("The movie was okay.", "neutral"),
    ("Not good at all.", "negative"),
    ("Absolutely fantastic!", "positive"),
    ("I don't really like it.", "negative"),
    ("Pretty decent movie.", "positive"),
    ("Quite boring film.", "negative")
]

rule_sentiment.rule_performance_analysis(labeled_test_data)
```

## 3. Machine Learning Approaches {#ml-approaches}

Machine learning approaches learn sentiment patterns from labeled training data.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

class MLSentimentAnalysis:
    """Machine Learning approaches for sentiment analysis"""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        
    def prepare_data(self):
        """Prepare sample dataset for training"""
        # Sample dataset (in practice, use larger datasets like IMDB, Amazon reviews, etc.)
        positive_reviews = [
            "I love this product! Amazing quality and fast shipping.",
            "Excellent service, highly recommended to everyone.",
            "Best purchase I've made this year. Outstanding!",
            "Great value for money, very satisfied with my order.",
            "Fantastic quality, exceeded my expectations completely.",
            "Perfect item, exactly what I was looking for.",
            "Wonderful experience, will definitely buy again.",
            "Superb quality and excellent customer service.",
            "I'm really happy with this purchase, great product.",
            "Amazing deal, product arrived quickly and works perfectly."
        ]
        
        negative_reviews = [
            "Terrible product, waste of money. Very disappointed.",
            "Poor quality, broke after one day of use.",
            "Worst purchase ever, completely useless item.",
            "Bad service, item arrived damaged and late.",
            "Horrible quality, nothing like the description.",
            "Disappointing product, definitely not worth the price.",
            "Awful experience, customer service was terrible.",
            "Low quality materials, product fell apart quickly.",
            "Regret buying this, complete waste of time and money.",
            "Very poor quality, would not recommend to anyone."
        ]
        
        neutral_reviews = [
            "The product is okay, nothing special about it.",
            "Average quality, does what it's supposed to do.",
            "It's fine, meets basic requirements but not impressive.",
            "Decent product for the price, acceptable quality.",
            "Standard item, neither good nor bad.",
            "Mediocre quality, could be better for the price.",
            "It works as expected, nothing more nothing less.",
            "Acceptable product, serves its purpose adequately.",
            "Fair quality, meets minimum requirements.",
            "Ordinary product, not exciting but functional."
        ]
        
        # Combine data
        texts = positive_reviews + negative_reviews + neutral_reviews
        labels = ['positive'] * len(positive_reviews) + ['negative'] * len(negative_reviews) + ['neutral'] * len(neutral_reviews)
        
        return texts, labels
    
    def feature_extraction_comparison(self, texts, labels):
        """Compare different feature extraction methods"""
        print("=== FEATURE EXTRACTION COMPARISON ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)
        
        # Different vectorizers
        vectorizers = {
            'Count': CountVectorizer(max_features=1000, stop_words='english'),
            'TF-IDF': TfidfVectorizer(max_features=1000, stop_words='english'),
            'Count (bigrams)': CountVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english'),
            'TF-IDF (bigrams)': TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
        }
        
        results = {}
        
        for name, vectorizer in vectorizers.items():
            # Transform features
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            
            # Train Naive Bayes classifier
            nb = MultinomialNB()
            nb.fit(X_train_vec, y_train)
            
            # Predict and evaluate
            y_pred = nb.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = accuracy
            
        # Display results
        print("Vectorizer Performance (Naive Bayes):")
        for name, accuracy in results.items():
            print(f"  {name:<20}: {accuracy:.3f}")
        
        return results
    
    def compare_ml_algorithms(self, texts, labels):
        """Compare different ML algorithms for sentiment analysis"""
        print("\n=== ML ALGORITHMS COMPARISON ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Different algorithms
        algorithms = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        results = {}
        
        for name, model in algorithms.items():
            # Train model
            model.fit(X_train_vec, y_train)
            
            # Predict
            y_pred = model.predict(X_test_vec)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'accuracy': accuracy,
                'model': model,
                'predictions': y_pred
            }
        
        # Display results
        print("Algorithm Performance:")
        for name, result in results.items():
            print(f"  {name:<20}: {result['accuracy']:.3f}")
        
        # Detailed evaluation for best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_predictions = results[best_model_name]['predictions']
        
        print(f"\nDetailed evaluation for {best_model_name}:")
        print(classification_report(y_test, best_predictions))
        
        return results, vectorizer
    
    def feature_importance_analysis(self, model, vectorizer, top_n=10):
        """Analyze feature importance for interpretability"""
        print(f"\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        if hasattr(model, 'coef_'):
            # For linear models like Logistic Regression
            feature_names = vectorizer.get_feature_names_out()
            
            # Get coefficients for each class
            if len(model.classes_) == 2:
                coefficients = model.coef_[0]
                
                # Most positive features
                top_positive_idx = np.argsort(coefficients)[-top_n:][::-1]
                print("Most Positive Features:")
                for i in top_positive_idx:
                    print(f"  {feature_names[i]}: {coefficients[i]:.3f}")
                
                # Most negative features
                top_negative_idx = np.argsort(coefficients)[:top_n]
                print("\nMost Negative Features:")
                for i in top_negative_idx:
                    print(f"  {feature_names[i]}: {coefficients[i]:.3f}")
            
            else:
                # Multi-class case
                for i, class_name in enumerate(model.classes_):
                    coefficients = model.coef_[i]
                    top_features_idx = np.argsort(coefficients)[-top_n:][::-1]
                    
                    print(f"\nTop features for '{class_name}':")
                    for idx in top_features_idx:
                        print(f"  {feature_names[idx]}: {coefficients[idx]:.3f}")
        
        elif hasattr(model, 'feature_importances_'):
            # For tree-based models
            feature_names = vectorizer.get_feature_names_out()
            importances = model.feature_importances_
            
            top_features_idx = np.argsort(importances)[-top_n:][::-1]
            
            print("Most Important Features:")
            for idx in top_features_idx:
                print(f"  {feature_names[idx]}: {importances[idx]:.3f}")
    
    def create_sentiment_pipeline(self, texts, labels):
        """Create an end-to-end sentiment analysis pipeline"""
        print("\n=== SENTIMENT ANALYSIS PIPELINE ===")
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)
        
        # Train pipeline
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Pipeline Accuracy: {accuracy:.3f}")
        
        # Test on new examples
        new_texts = [
            "This product is absolutely amazing!",
            "Terrible quality, very disappointed.",
            "It's okay, nothing special.",
            "Love it! Best purchase ever!",
            "Poor service, would not recommend."
        ]
        
        predictions = pipeline.predict(new_texts)
        probabilities = pipeline.predict_proba(new_texts)
        
        print("\nPredictions on new texts:")
        print(f"{'Text':<40} {'Prediction':<12} {'Confidence'}")
        print("-" * 65)
        
        for text, pred, prob in zip(new_texts, predictions, probabilities):
            confidence = max(prob)
            short_text = text[:35] + "..." if len(text) > 35 else text
            print(f"{short_text:<40} {pred:<12} {confidence:.3f}")
        
        return pipeline

# Demonstrate ML sentiment analysis
ml_sentiment = MLSentimentAnalysis()

# Prepare data
texts, labels = ml_sentiment.prepare_data()

# Compare feature extraction methods
ml_sentiment.feature_extraction_comparison(texts, labels)

# Compare ML algorithms
results, vectorizer = ml_sentiment.compare_ml_algorithms(texts, labels)

# Analyze feature importance
best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
best_model = results[best_model_name]['model']
ml_sentiment.feature_importance_analysis(best_model, vectorizer)

# Create complete pipeline
pipeline = ml_sentiment.create_sentiment_pipeline(texts, labels)
```

## 4. Deep Learning for Sentiment Analysis {#deep-learning}

Deep learning models can capture complex patterns and contextual relationships in text.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

class DeepLearningSentiment:
    """Deep learning approaches for sentiment analysis"""
    
    def __init__(self, max_vocab_size=10000, max_sequence_length=100):
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.tokenizer = Tokenizer(num_words=max_vocab_size)
        self.models = {}
    
    def prepare_deep_learning_data(self, texts, labels):
        """Prepare data for deep learning models"""
        print("=== PREPARING DATA FOR DEEP LEARNING ===")
        
        # Fit tokenizer
        self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        X = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        
        # Encode labels
        label_encoder = {'positive': 2, 'neutral': 1, 'negative': 0}
        y = np.array([label_encoder[label] for label in labels])
        y_categorical = to_categorical(y, num_classes=3)
        
        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"Sequence shape: {X.shape}")
        print(f"Labels shape: {y_categorical.shape}")
        
        return X, y_categorical, label_encoder
    
    def build_simple_lstm(self, num_classes=3):
        """Build a simple LSTM model"""
        model = Sequential([
            Embedding(self.max_vocab_size, 128, input_length=self.max_sequence_length),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_bidirectional_lstm(self, num_classes=3):
        """Build a bidirectional LSTM model"""
        model = Sequential([
            Embedding(self.max_vocab_size, 128, input_length=self.max_sequence_length),
            Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_stacked_lstm(self, num_classes=3):
        """Build a stacked LSTM model"""
        model = Sequential([
            Embedding(self.max_vocab_size, 128, input_length=self.max_sequence_length),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
            LSTM(32, dropout=0.2, recurrent_dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_and_compare_models(self, X, y, validation_split=0.2, epochs=10):
        """Train and compare different deep learning models"""
        print("\n=== TRAINING DEEP LEARNING MODELS ===")
        
        # Define models
        model_builders = {
            'Simple LSTM': self.build_simple_lstm,
            'Bidirectional LSTM': self.build_bidirectional_lstm,
            'Stacked LSTM': self.build_stacked_lstm
        }
        
        results = {}
        
        for name, builder in model_builders.items():
            print(f"\nTraining {name}...")
            
            # Build model
            model = builder()
            
            # Train model
            history = model.fit(
                X, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=32,
                verbose=0
            )
            
            # Store results
            final_accuracy = history.history['accuracy'][-1]
            final_val_accuracy = history.history['val_accuracy'][-1]
            
            results[name] = {
                'model': model,
                'history': history,
                'train_accuracy': final_accuracy,
                'val_accuracy': final_val_accuracy
            }
            
            print(f"  Final training accuracy: {final_accuracy:.3f}")
            print(f"  Final validation accuracy: {final_val_accuracy:.3f}")
        
        return results
    
    def visualize_training_history(self, results):
        """Visualize training history for all models"""
        print("\n=== TRAINING HISTORY VISUALIZATION ===")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        for name, result in results.items():
            history = result['history']
            
            # Plot training & validation accuracy
            axes[0].plot(history.history['accuracy'], label=f'{name} (Train)')
            axes[0].plot(history.history['val_accuracy'], label=f'{name} (Val)', linestyle='--')
            
            # Plot training & validation loss
            axes[1].plot(history.history['loss'], label=f'{name} (Train)')
            axes[1].plot(history.history['val_loss'], label=f'{name} (Val)', linestyle='--')
        
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict_sentiment(self, model, texts, label_encoder):
        """Predict sentiment for new texts"""
        # Reverse label encoder
        reverse_encoder = {v: k for k, v in label_encoder.items()}
        
        # Preprocess texts
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        
        # Predict
        predictions = model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        results = []
        for i, text in enumerate(texts):
            sentiment = reverse_encoder[predicted_classes[i]]
            confidence = confidence_scores[i]
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence
            })
        
        return results

# Example usage (commented out as it requires substantial data and training time)
"""
# Demonstrate deep learning sentiment analysis
dl_sentiment = DeepLearningSentiment()

# Prepare larger dataset (you would use real datasets like IMDB here)
texts, labels = ml_sentiment.prepare_data()

# Expand dataset for deep learning (normally you'd have thousands of examples)
expanded_texts = texts * 10  # Simple expansion for demo
expanded_labels = labels * 10

X, y, label_encoder = dl_sentiment.prepare_deep_learning_data(expanded_texts, expanded_labels)

# Train models
results = dl_sentiment.train_and_compare_models(X, y, epochs=5)

# Visualize training
dl_sentiment.visualize_training_history(results)

# Test predictions
best_model_name = max(results.keys(), key=lambda k: results[k]['val_accuracy'])
best_model = results[best_model_name]['model']

test_texts = [
    "This movie is absolutely fantastic!",
    "Terrible film, waste of time.",
    "It was okay, nothing special."
]

predictions = dl_sentiment.predict_sentiment(best_model, test_texts, label_encoder)

print(f"\nPredictions using {best_model_name}:")
for pred in predictions:
    print(f"Text: {pred['text']}")
    print(f"Sentiment: {pred['sentiment']} (confidence: {pred['confidence']:.3f})")
    print()
"""

print("Deep Learning Sentiment Analysis module ready (commented out due to training time)")
```

## Summary

This chapter covered comprehensive approaches to sentiment analysis:

1. **Introduction**: Understanding different types and applications of sentiment analysis
2. **Rule-Based Approaches**: Using lexicons and linguistic rules for sentiment detection
3. **Machine Learning**: Leveraging supervised learning with feature engineering
4. **Deep Learning**: Advanced neural network architectures for sentiment classification

### Key Takeaways:
- Rule-based methods are interpretable but limited by lexicon coverage
- Machine learning approaches require labeled data but can generalize better
- Deep learning models can capture complex patterns but need more data and computation
- Feature engineering is crucial for traditional ML approaches
- Context and negation handling significantly improve performance
- Evaluation should consider real-world performance, not just accuracy

### Best Practices:
- Start with rule-based approaches for quick prototypes
- Use machine learning for better generalization
- Consider deep learning for complex tasks with sufficient data
- Handle negation, intensifiers, and context appropriately
- Evaluate on domain-specific data
- Consider multi-class and aspect-based sentiment when relevant

---

## Exercises

1. **Lexicon Enhancement**: Build a domain-specific sentiment lexicon
2. **Feature Engineering**: Experiment with different feature extraction techniques
3. **Model Comparison**: Compare all approaches on the same dataset
4. **Real-world Application**: Build a sentiment analyzer for product reviews
5. **Advanced Techniques**: Implement attention mechanisms for deep learning models

---

*Sentiment analysis is a fundamental NLP task with wide applications. Master these techniques to build effective opinion mining systems.* 