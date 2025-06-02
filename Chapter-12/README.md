# Chapter 12: Natural Language Processing Essentials

## Learning Objectives
By the end of this chapter, students will be able to:
- Understand fundamental concepts in Natural Language Processing (NLP)
- Apply text preprocessing and tokenization techniques
- Implement various NLP algorithms and models
- Build text classification and sentiment analysis systems
- Evaluate NLP model performance

## Table of Contents
1. [Introduction to NLP](#introduction)
2. [Text Preprocessing](#preprocessing)
3. [Tokenization and Text Representation](#tokenization)
4. [Language Models](#language-models)
5. [Text Classification](#classification)
6. [Named Entity Recognition](#ner)
7. [Part-of-Speech Tagging](#pos)
8. [Word Embeddings](#embeddings)
9. [Sequence-to-Sequence Models](#seq2seq)
10. [Evaluation Metrics](#evaluation)

## 1. Introduction to NLP {#introduction}

Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate human language in a meaningful way.

### Key NLP Tasks:
- **Text Classification**: Categorizing text into predefined classes
- **Sentiment Analysis**: Determining emotional tone
- **Named Entity Recognition**: Identifying entities in text
- **Machine Translation**: Converting text between languages
- **Question Answering**: Providing answers to questions
- **Text Summarization**: Creating concise summaries

### NLP Pipeline:
1. **Text Acquisition**: Gathering raw text data
2. **Text Preprocessing**: Cleaning and normalizing text
3. **Feature Extraction**: Converting text to numerical features
4. **Model Training**: Training ML models on processed data
5. **Evaluation**: Assessing model performance
6. **Deployment**: Implementing models in applications

## 2. Text Preprocessing {#preprocessing}

```python
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class TextPreprocessor:
    def __init__(self):
        """Initialize text preprocessor"""
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text):
        """Remove stopwords from text"""
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(filtered_tokens)
    
    def stem_text(self, text):
        """Apply stemming to text"""
        tokens = word_tokenize(text)
        stemmed_tokens = [self.stemmer.stem(word) for word in tokens]
        return ' '.join(stemmed_tokens)
    
    def lemmatize_text(self, text):
        """Apply lemmatization to text"""
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(lemmatized_tokens)
    
    def preprocess(self, text, remove_stopwords=True, use_stemming=False, use_lemmatization=True):
        """Complete preprocessing pipeline"""
        # Clean text
        text = self.clean_text(text)
        
        # Remove stopwords
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        # Apply stemming or lemmatization
        if use_stemming:
            text = self.stem_text(text)
        elif use_lemmatization:
            text = self.lemmatize_text(text)
        
        return text

# Usage example
preprocessor = TextPreprocessor()
sample_text = "Hello! This is a sample text with URLs: https://example.com and @mentions #hashtags"
processed_text = preprocessor.preprocess(sample_text)
print(f"Original: {sample_text}")
print(f"Processed: {processed_text}")
```

## 3. Tokenization and Text Representation {#tokenization}

### Word-Level Tokenization:

```python
from collections import Counter
import numpy as np

class VocabularyBuilder:
    def __init__(self, max_vocab_size=10000, min_freq=2):
        """Initialize vocabulary builder"""
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_frequencies = Counter()
    
    def build_vocabulary(self, texts):
        """Build vocabulary from text corpus"""
        # Count word frequencies
        for text in texts:
            words = text.split()
            self.word_frequencies.update(words)
        
        # Filter words by frequency
        filtered_words = [word for word, freq in self.word_frequencies.items() 
                         if freq >= self.min_freq]
        
        # Sort by frequency and limit vocabulary size
        most_common = self.word_frequencies.most_common(self.max_vocab_size)
        vocab_words = [word for word, _ in most_common if word in filtered_words]
        
        # Create mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # Add special tokens
        self.word_to_idx['<UNK>'] = len(self.word_to_idx)
        self.word_to_idx['<PAD>'] = len(self.word_to_idx)
        self.idx_to_word[len(self.idx_to_word)] = '<UNK>'
        self.idx_to_word[len(self.idx_to_word)] = '<PAD>'
    
    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        words = text.split()
        sequence = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
        return sequence
    
    def sequence_to_text(self, sequence):
        """Convert sequence of indices to text"""
        words = [self.idx_to_word.get(idx, '<UNK>') for idx in sequence]
        return ' '.join(words)
```

### Bag of Words (BoW):

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class BagOfWords:
    def __init__(self, max_features=1000, use_tfidf=False):
        """Initialize Bag of Words model"""
        self.max_features = max_features
        self.use_tfidf = use_tfidf
        
        if use_tfidf:
            self.vectorizer = TfidfVectorizer(max_features=max_features)
        else:
            self.vectorizer = CountVectorizer(max_features=max_features)
    
    def fit_transform(self, texts):
        """Fit vectorizer and transform texts"""
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        """Transform texts using fitted vectorizer"""
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Get feature names (words)"""
        return self.vectorizer.get_feature_names_out()

# Example usage
texts = [
    "I love machine learning",
    "Natural language processing is fascinating",
    "Deep learning models are powerful"
]

bow = BagOfWords(use_tfidf=True)
tfidf_matrix = bow.fit_transform(texts)
print("TF-IDF Matrix shape:", tfidf_matrix.shape)
print("Feature names:", bow.get_feature_names()[:10])
```

### N-grams:

```python
from sklearn.feature_extraction.text import CountVectorizer

def extract_ngrams(text, n=2):
    """Extract n-grams from text"""
    words = text.split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    return ngrams

class NgramExtractor:
    def __init__(self, n_range=(1, 2), max_features=1000):
        """Initialize n-gram extractor"""
        self.vectorizer = CountVectorizer(
            ngram_range=n_range,
            max_features=max_features
        )
    
    def fit_transform(self, texts):
        """Extract n-grams from texts"""
        return self.vectorizer.fit_transform(texts)
    
    def get_feature_names(self):
        """Get n-gram features"""
        return self.vectorizer.get_feature_names_out()

# Example
ngram_extractor = NgramExtractor(n_range=(1, 3))
ngram_matrix = ngram_extractor.fit_transform(texts)
print("N-gram features:", ngram_extractor.get_feature_names()[:10])
```

## 4. Language Models {#language-models}

### N-gram Language Model:

```python
from collections import defaultdict, Counter
import random

class NgramLanguageModel:
    def __init__(self, n=2):
        """Initialize n-gram language model"""
        self.n = n
        self.ngrams = defaultdict(Counter)
        self.vocabulary = set()
    
    def train(self, texts):
        """Train n-gram model on text corpus"""
        for text in texts:
            words = ['<START>'] * (self.n - 1) + text.split() + ['<END>']
            self.vocabulary.update(words)
            
            # Extract n-grams
            for i in range(len(words) - self.n + 1):
                context = tuple(words[i:i+self.n-1])
                next_word = words[i+self.n-1]
                self.ngrams[context][next_word] += 1
    
    def get_probability(self, context, word):
        """Get probability of word given context"""
        context = tuple(context)
        if context not in self.ngrams:
            return 0.0
        
        total_count = sum(self.ngrams[context].values())
        word_count = self.ngrams[context][word]
        return word_count / total_count if total_count > 0 else 0.0
    
    def generate_text(self, length=10, seed=None):
        """Generate text using the language model"""
        if seed:
            random.seed(seed)
        
        # Start with initial context
        context = ['<START>'] * (self.n - 1)
        generated = []
        
        for _ in range(length):
            context_tuple = tuple(context)
            if context_tuple not in self.ngrams:
                break
            
            # Sample next word based on probabilities
            candidates = list(self.ngrams[context_tuple].keys())
            weights = list(self.ngrams[context_tuple].values())
            
            if not candidates:
                break
            
            next_word = random.choices(candidates, weights=weights)[0]
            
            if next_word == '<END>':
                break
            
            generated.append(next_word)
            context = context[1:] + [next_word]
        
        return ' '.join(generated)

# Example usage
texts = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "cats and dogs are pets"
]

lm = NgramLanguageModel(n=2)
lm.train(texts)
generated_text = lm.generate_text(length=8)
print("Generated text:", generated_text)
```

## 5. Text Classification {#classification}

### Naive Bayes Classifier:

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class TextClassifier:
    def __init__(self, vectorizer_type='tfidf'):
        """Initialize text classifier"""
        self.vectorizer_type = vectorizer_type
        self.classifier = MultinomialNB()
        
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=5000)
        else:
            self.vectorizer = CountVectorizer(max_features=5000)
    
    def train(self, texts, labels):
        """Train the text classifier"""
        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)
        
        # Train classifier
        self.classifier.fit(X, labels)
    
    def predict(self, texts):
        """Predict labels for texts"""
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)
    
    def predict_proba(self, texts):
        """Predict probabilities for texts"""
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)
    
    def evaluate(self, texts, labels):
        """Evaluate classifier performance"""
        predictions = self.predict(texts)
        accuracy = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions)
        return accuracy, report

# Example: Sentiment Classification
sentiment_texts = [
    "I love this movie! It's amazing!",
    "This film is terrible and boring.",
    "Great acting and wonderful story.",
    "Worst movie I've ever seen.",
    "Fantastic cinematography and direction."
]
sentiment_labels = ['positive', 'negative', 'positive', 'negative', 'positive']

classifier = TextClassifier()
classifier.train(sentiment_texts, sentiment_labels)

test_texts = ["This movie is wonderful!", "I hate this film."]
predictions = classifier.predict(test_texts)
print("Predictions:", predictions)
```

### Deep Learning Text Classification:

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DeepTextClassifier:
    def __init__(self, max_vocab_size=10000, max_sequence_length=100, embedding_dim=100):
        """Initialize deep text classifier"""
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_vocab_size)
        self.model = None
    
    def preprocess_texts(self, texts):
        """Preprocess texts for training"""
        # Fit tokenizer
        self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding='post'
        )
        
        return padded_sequences
    
    def build_model(self, num_classes):
        """Build deep learning model"""
        model = models.Sequential([
            layers.Embedding(
                self.max_vocab_size,
                self.embedding_dim,
                input_length=self.max_sequence_length
            ),
            layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, texts, labels, validation_split=0.2, epochs=10):
        """Train the deep classifier"""
        # Preprocess texts
        X = self.preprocess_texts(texts)
        
        # Build model
        num_classes = len(set(labels))
        self.build_model(num_classes)
        
        # Train model
        history = self.model.fit(
            X, labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        return history
    
    def predict(self, texts):
        """Predict labels for texts"""
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding='post'
        )
        
        predictions = self.model.predict(padded_sequences)
        return predictions.argmax(axis=1)
```

## 6. Named Entity Recognition {#ner}

```python
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

class NERModel:
    def __init__(self, model_name='en_core_web_sm'):
        """Initialize NER model"""
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            self.nlp = spacy.blank('en')
            self.nlp.add_pipe('ner')
        
        self.ner = self.nlp.get_pipe('ner')
    
    def add_labels(self, labels):
        """Add entity labels to the model"""
        for label in labels:
            self.ner.add_label(label)
    
    def train(self, training_data, epochs=10):
        """Train NER model"""
        # Convert training data to spaCy format
        examples = []
        for text, annotations in training_data:
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        
        # Train the model
        optimizer = self.nlp.resume_training()
        
        for epoch in range(epochs):
            random.shuffle(examples)
            losses = {}
            
            # Create mini-batches
            batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
            
            for batch in batches:
                self.nlp.update(batch, losses=losses, drop=0.2)
            
            print(f"Epoch {epoch + 1}, Losses: {losses}")
    
    def predict(self, text):
        """Extract entities from text"""
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_, ent.start, ent.end) for ent in doc.ents]
        return entities
    
    def evaluate(self, test_data):
        """Evaluate NER model"""
        correct = 0
        total = 0
        
        for text, annotations in test_data:
            predicted_entities = self.predict(text)
            true_entities = annotations.get('entities', [])
            
            for entity in true_entities:
                total += 1
                start, end, label = entity
                entity_text = text[start:end]
                
                # Check if predicted correctly
                for pred_text, pred_label, pred_start, pred_end in predicted_entities:
                    if (pred_start == start and pred_end == end and 
                        pred_label == label):
                        correct += 1
                        break
        
        accuracy = correct / total if total > 0 else 0
        return accuracy

# Example training data
training_data = [
    ("Apple Inc. is based in Cupertino.", {"entities": [(0, 9, "ORG"), (22, 31, "GPE")]}),
    ("John works at Microsoft in Seattle.", {"entities": [(0, 4, "PERSON"), (14, 23, "ORG"), (27, 34, "GPE")]}),
    ("Google was founded by Larry Page.", {"entities": [(0, 6, "ORG"), (22, 32, "PERSON")]})
]

ner_model = NERModel()
ner_model.add_labels(["PERSON", "ORG", "GPE"])
ner_model.train(training_data)

# Test the model
test_text = "Tim Cook leads Apple in California."
entities = ner_model.predict(test_text)
print("Entities:", entities)
```

## 7. Part-of-Speech Tagging {#pos}

```python
import nltk
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

class POSTagger:
    def __init__(self):
        """Initialize POS tagger"""
        self.vectorizer = DictVectorizer()
        self.classifier = LogisticRegression()
        self.tag_to_idx = {}
        self.idx_to_tag = {}
    
    def word_features(self, sentence, index):
        """Extract features for a word"""
        word = sentence[index]
        features = {
            'word': word.lower(),
            'is_first': index == 0,
            'is_last': index == len(sentence) - 1,
            'is_capitalized': word[0].upper() == word[0],
            'is_all_caps': word.upper() == word,
            'is_all_lower': word.lower() == word,
            'prefix-1': word[0] if len(word) > 0 else '',
            'prefix-2': word[:2] if len(word) > 1 else '',
            'prefix-3': word[:3] if len(word) > 2 else '',
            'suffix-1': word[-1] if len(word) > 0 else '',
            'suffix-2': word[-2:] if len(word) > 1 else '',
            'suffix-3': word[-3:] if len(word) > 2 else '',
            'prev_word': '' if index == 0 else sentence[index-1].lower(),
            'next_word': '' if index == len(sentence)-1 else sentence[index+1].lower(),
            'has_hyphen': '-' in word,
            'is_numeric': word.isdigit(),
            'capitals_inside': word[1:].lower() != word[1:]
        }
        return features
    
    def prepare_data(self, tagged_sentences):
        """Prepare training data"""
        X, y = [], []
        
        # Build tag vocabulary
        all_tags = set()
        for sentence in tagged_sentences:
            for word, tag in sentence:
                all_tags.add(tag)
        
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(all_tags)}
        self.idx_to_tag = {idx: tag for tag, idx in self.tag_to_idx.items()}
        
        # Extract features
        for sentence in tagged_sentences:
            words = [word for word, tag in sentence]
            for index, (word, tag) in enumerate(sentence):
                features = self.word_features(words, index)
                X.append(features)
                y.append(self.tag_to_idx[tag])
        
        return X, y
    
    def train(self, tagged_sentences):
        """Train POS tagger"""
        X, y = self.prepare_data(tagged_sentences)
        
        # Vectorize features
        X_vectorized = self.vectorizer.fit_transform(X)
        
        # Train classifier
        self.classifier.fit(X_vectorized, y)
    
    def tag_sentence(self, sentence):
        """Tag a sentence with POS tags"""
        words = sentence.split()
        tagged = []
        
        for index, word in enumerate(words):
            features = self.word_features(words, index)
            features_vectorized = self.vectorizer.transform([features])
            tag_idx = self.classifier.predict(features_vectorized)[0]
            tag = self.idx_to_tag[tag_idx]
            tagged.append((word, tag))
        
        return tagged

# Example usage with NLTK's tagged corpus
from nltk.corpus import treebank

# Load sample data
tagged_sentences = treebank.tagged_sents()[:1000]

# Train custom POS tagger
pos_tagger = POSTagger()
pos_tagger.train(tagged_sentences)

# Test the tagger
test_sentence = "The quick brown fox jumps over the lazy dog"
tagged_result = pos_tagger.tag_sentence(test_sentence)
print("POS Tags:", tagged_result)
```

## 8. Word Embeddings {#embeddings}

### Word2Vec Implementation:

```python
import numpy as np
from collections import defaultdict, Counter
import random

class Word2Vec:
    def __init__(self, vector_size=100, window=5, min_count=1, epochs=5):
        """Initialize Word2Vec model"""
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.embeddings = None
    
    def build_vocabulary(self, sentences):
        """Build vocabulary from sentences"""
        word_counts = Counter()
        
        for sentence in sentences:
            words = sentence.split()
            word_counts.update(words)
        
        # Filter by minimum count
        vocab_words = [word for word, count in word_counts.items() 
                      if count >= self.min_count]
        
        # Create mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(vocab_words)
    
    def generate_training_data(self, sentences):
        """Generate skip-gram training pairs"""
        training_data = []
        
        for sentence in sentences:
            words = sentence.split()
            for i, target_word in enumerate(words):
                if target_word not in self.word_to_idx:
                    continue
                
                target_idx = self.word_to_idx[target_word]
                
                # Get context words
                start = max(0, i - self.window)
                end = min(len(words), i + self.window + 1)
                
                for j in range(start, end):
                    if i != j and words[j] in self.word_to_idx:
                        context_idx = self.word_to_idx[words[j]]
                        training_data.append((target_idx, context_idx))
        
        return training_data
    
    def train(self, sentences, learning_rate=0.01):
        """Train Word2Vec model using skip-gram"""
        # Build vocabulary
        self.build_vocabulary(sentences)
        
        # Initialize embeddings
        self.embeddings = np.random.uniform(
            -0.5/self.vector_size, 0.5/self.vector_size,
            (self.vocab_size, self.vector_size)
        )
        
        # Generate training data
        training_data = self.generate_training_data(sentences)
        
        # Training loop
        for epoch in range(self.epochs):
            random.shuffle(training_data)
            total_loss = 0
            
            for target_idx, context_idx in training_data:
                # Simple skip-gram implementation
                # This is a simplified version - actual Word2Vec uses hierarchical softmax
                loss = self.train_pair(target_idx, context_idx, learning_rate)
                total_loss += loss
            
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(training_data)}")
    
    def train_pair(self, target_idx, context_idx, learning_rate):
        """Train on a single word pair"""
        # Get embeddings
        target_embedding = self.embeddings[target_idx]
        context_embedding = self.embeddings[context_idx]
        
        # Compute dot product
        dot_product = np.dot(target_embedding, context_embedding)
        
        # Sigmoid activation
        prediction = 1 / (1 + np.exp(-dot_product))
        
        # Compute loss (binary cross-entropy)
        loss = -np.log(prediction + 1e-10)
        
        # Compute gradients
        gradient = (prediction - 1) * learning_rate
        
        # Update embeddings
        self.embeddings[target_idx] -= gradient * context_embedding
        self.embeddings[context_idx] -= gradient * target_embedding
        
        return loss
    
    def get_vector(self, word):
        """Get vector for a word"""
        if word in self.word_to_idx:
            idx = self.word_to_idx[word]
            return self.embeddings[idx]
        return None
    
    def similarity(self, word1, word2):
        """Compute cosine similarity between two words"""
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)
        
        if vec1 is not None and vec2 is not None:
            cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return cosine_sim
        return None
    
    def most_similar(self, word, top_n=5):
        """Find most similar words"""
        if word not in self.word_to_idx:
            return []
        
        word_vector = self.get_vector(word)
        similarities = []
        
        for other_word in self.word_to_idx:
            if other_word != word:
                sim = self.similarity(word, other_word)
                if sim is not None:
                    similarities.append((other_word, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

# Example usage
sentences = [
    "the cat sat on the mat",
    "cats and dogs are pets",
    "I love my pet cat",
    "dogs are loyal animals",
    "animals are wonderful creatures"
]

word2vec = Word2Vec(vector_size=50, epochs=10)
word2vec.train(sentences)

# Test similarity
similarity = word2vec.similarity("cat", "cats")
print(f"Similarity between 'cat' and 'cats': {similarity}")

# Find similar words
similar_words = word2vec.most_similar("cat", top_n=3)
print(f"Words similar to 'cat': {similar_words}")
```

## 9. Sequence-to-Sequence Models {#seq2seq}

```python
import tensorflow as tf
from tensorflow.keras import layers, models

class Seq2SeqModel:
    def __init__(self, input_vocab_size, target_vocab_size, embedding_dim=256, hidden_units=512):
        """Initialize Seq2Seq model"""
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
    
    def _build_encoder(self):
        """Build encoder network"""
        encoder_inputs = layers.Input(shape=(None,))
        encoder_embedding = layers.Embedding(self.input_vocab_size, self.embedding_dim)(encoder_inputs)
        encoder_lstm = layers.LSTM(self.hidden_units, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]
        
        return models.Model(encoder_inputs, encoder_states)
    
    def _build_decoder(self):
        """Build decoder network"""
        decoder_inputs = layers.Input(shape=(None,))
        decoder_embedding = layers.Embedding(self.target_vocab_size, self.embedding_dim)
        decoder_embedding_output = decoder_embedding(decoder_inputs)
        
        decoder_lstm = layers.LSTM(self.hidden_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding_output, initial_state=None)
        
        decoder_dense = layers.Dense(self.target_vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        return models.Model([decoder_inputs], decoder_outputs)
    
    def build_training_model(self):
        """Build model for training"""
        # Encoder
        encoder_inputs = layers.Input(shape=(None,))
        encoder_embedding = layers.Embedding(self.input_vocab_size, self.embedding_dim)(encoder_inputs)
        encoder_lstm = layers.LSTM(self.hidden_units, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = layers.Input(shape=(None,))
        decoder_embedding = layers.Embedding(self.target_vocab_size, self.embedding_dim)
        decoder_embedding_output = decoder_embedding(decoder_inputs)
        
        decoder_lstm = layers.LSTM(self.hidden_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding_output, initial_state=encoder_states)
        
        decoder_dense = layers.Dense(self.target_vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        return model
    
    def train(self, input_sequences, target_sequences, epochs=50):
        """Train the Seq2Seq model"""
        # Build training model
        model = self.build_training_model()
        
        # Compile model
        model.compile(
            optimizer='rmsprop',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Prepare decoder input (shifted target sequences)
        decoder_input_data = target_sequences[:, :-1]
        decoder_target_data = target_sequences[:, 1:]
        
        # Train model
        history = model.fit(
            [input_sequences, decoder_input_data],
            decoder_target_data,
            batch_size=64,
            epochs=epochs,
            validation_split=0.2,
            verbose=1
        )
        
        return history
```

## 10. Evaluation Metrics {#evaluation}

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

class NLPEvaluator:
    def __init__(self):
        """Initialize NLP evaluator"""
        pass
    
    def classification_metrics(self, y_true, y_pred, labels=None):
        """Compute classification metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support
        }
    
    def bleu_score(self, reference, candidate):
        """Compute BLEU score for text generation"""
        # Simplified BLEU implementation
        ref_words = reference.split()
        cand_words = candidate.split()
        
        if len(cand_words) == 0:
            return 0.0
        
        # Compute n-gram precisions
        precisions = []
        for n in range(1, 5):  # 1-gram to 4-gram
            ref_ngrams = self._get_ngrams(ref_words, n)
            cand_ngrams = self._get_ngrams(cand_words, n)
            
            if len(cand_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            matches = 0
            for ngram in cand_ngrams:
                if ngram in ref_ngrams:
                    matches += min(cand_ngrams[ngram], ref_ngrams[ngram])
            
            precision = matches / sum(cand_ngrams.values())
            precisions.append(precision)
        
        # Compute geometric mean
        if any(p == 0 for p in precisions):
            return 0.0
        
        bleu = np.exp(np.mean(np.log(precisions)))
        
        # Brevity penalty
        bp = 1.0 if len(cand_words) >= len(ref_words) else np.exp(1 - len(ref_words) / len(cand_words))
        
        return bp * bleu
    
    def _get_ngrams(self, words, n):
        """Get n-grams from word list"""
        ngrams = {}
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams
    
    def rouge_score(self, reference, candidate):
        """Compute ROUGE score for summarization"""
        ref_words = set(reference.split())
        cand_words = set(candidate.split())
        
        if len(ref_words) == 0:
            return 0.0
        
        overlap = len(ref_words.intersection(cand_words))
        rouge = overlap / len(ref_words)
        
        return rouge
    
    def perplexity(self, probabilities):
        """Compute perplexity for language models"""
        log_probs = np.log(probabilities + 1e-10)
        avg_log_prob = np.mean(log_probs)
        perplexity = np.exp(-avg_log_prob)
        return perplexity

# Example usage
evaluator = NLPEvaluator()

# Classification metrics
y_true = ['positive', 'negative', 'positive', 'negative']
y_pred = ['positive', 'positive', 'positive', 'negative']
metrics = evaluator.classification_metrics(y_true, y_pred)
print("Classification metrics:", metrics)

# BLEU score
reference = "the cat sat on the mat"
candidate = "a cat was sitting on the mat"
bleu = evaluator.bleu_score(reference, candidate)
print(f"BLEU score: {bleu:.4f}")

# ROUGE score
rouge = evaluator.rouge_score(reference, candidate)
print(f"ROUGE score: {rouge:.4f}")
```

## Summary

This chapter covered essential NLP concepts and techniques:

1. **Text Preprocessing**: Cleaning and normalizing text data
2. **Tokenization**: Converting text to numerical representations
3. **Language Models**: Modeling probability distributions over text
4. **Text Classification**: Categorizing text into classes
5. **NER and POS Tagging**: Identifying entities and grammatical roles
6. **Word Embeddings**: Dense vector representations of words
7. **Seq2Seq Models**: Handling variable-length input-output pairs
8. **Evaluation**: Measuring NLP model performance

### Key Takeaways:
- Text preprocessing is crucial for NLP success
- Different representation methods suit different tasks
- Deep learning has revolutionized NLP performance
- Proper evaluation metrics are essential
- Transfer learning and pre-trained models are powerful

---

## Exercises

1. Build a complete sentiment analysis pipeline
2. Implement a custom tokenizer for a specific domain
3. Train word embeddings on domain-specific text
4. Create a named entity recognition system
5. Develop a text summarization model

---

*Master NLP through hands-on implementation of various techniques and models.* 