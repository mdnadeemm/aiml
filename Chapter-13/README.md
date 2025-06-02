# Chapter 13: Text Processing and Tokenization

## Learning Objectives
By the end of this chapter, students will be able to:
- Understand advanced text preprocessing techniques
- Implement various tokenization methods from scratch
- Apply text normalization and cleaning strategies
- Handle multilingual text processing challenges
- Build robust text processing pipelines

## Table of Contents
1. [Introduction to Text Processing](#introduction)
2. [Basic Text Cleaning](#basic-cleaning)
3. [Tokenization Techniques](#tokenization)
4. [Text Normalization](#normalization)
5. [Handling Special Cases](#special-cases)
6. [Multilingual Text Processing](#multilingual)
7. [Advanced Preprocessing](#advanced-preprocessing)
8. [Building Text Processing Pipelines](#pipelines)

## 1. Introduction to Text Processing {#introduction}

Text processing is the foundation of all NLP tasks. Raw text data is messy and inconsistent, requiring systematic cleaning and standardization before it can be effectively used in machine learning models.

### Why Text Processing Matters

```python
import re
import string
import unicodedata
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class TextProcessingDemo:
    """Demonstrate the importance of text processing"""
    
    def __init__(self):
        self.sample_texts = [
            "Hello World! This is a sample text.",
            "HELLO WORLD!!! this is ANOTHER sample text...",
            "hElLo WoRlD???    this    is   yet another sample!!!",
            "Hello, World! This text has émojis 😊 and spëcial characters.",
            "   hello world   this text has extra spaces   ",
            "Hello World! This text has\nnewlines\tand\ttabs.",
            "Hello World! Visit https://example.com for more info.",
            "Contact us at info@example.com or call (555) 123-4567."
        ]
    
    def demonstrate_raw_vs_processed(self):
        """Show difference between raw and processed text"""
        print("=== RAW vs PROCESSED TEXT ===")
        
        def basic_preprocess(text):
            # Convert to lowercase
            text = text.lower()
            # Remove punctuation
            text = re.sub(r'[^\w\s]', '', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text
        
        print("Raw texts:")
        for i, text in enumerate(self.sample_texts[:4], 1):
            print(f"{i}. {text}")
        
        print("\nProcessed texts:")
        for i, text in enumerate(self.sample_texts[:4], 1):
            processed = basic_preprocess(text)
            print(f"{i}. {processed}")
        
        # Vocabulary analysis
        raw_vocab = set()
        processed_vocab = set()
        
        for text in self.sample_texts:
            raw_vocab.update(text.split())
            processed_vocab.update(basic_preprocess(text).split())
        
        print(f"\nVocabulary comparison:")
        print(f"Raw vocabulary size: {len(raw_vocab)}")
        print(f"Processed vocabulary size: {len(processed_vocab)}")
        print(f"Reduction: {(1 - len(processed_vocab)/len(raw_vocab))*100:.1f}%")
        
        return raw_vocab, processed_vocab

demo = TextProcessingDemo()
demo.demonstrate_raw_vs_processed()
```

### Text Processing Challenges

```python
class TextChallenges:
    """Demonstrate common text processing challenges"""
    
    def encoding_issues(self):
        """Show encoding and decoding problems"""
        print("\n=== ENCODING CHALLENGES ===")
        
        # Different encodings
        text = "Café, naïve, résumé, 北京"
        
        encodings = ['utf-8', 'latin-1', 'ascii']
        
        for encoding in encodings:
            try:
                encoded = text.encode(encoding)
                decoded = encoded.decode(encoding)
                print(f"{encoding}: {decoded}")
            except UnicodeEncodeError as e:
                print(f"{encoding}: Error - {e}")
            except UnicodeDecodeError as e:
                print(f"{encoding}: Error - {e}")
    
    def inconsistent_formats(self):
        """Show format inconsistencies"""
        print("\n=== FORMAT INCONSISTENCIES ===")
        
        examples = [
            "Phone: 555-123-4567",
            "Phone: (555) 123-4567",
            "Phone: 555.123.4567",
            "Phone: +1-555-123-4567",
            "Call: 5551234567"
        ]
        
        print("Raw phone numbers:")
        for example in examples:
            print(f"  {example}")
        
        # Extract phone numbers with regex
        phone_pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        
        print("\nExtracted and normalized:")
        for example in examples:
            match = re.search(phone_pattern, example)
            if match:
                groups = match.groups()
                normalized = f"({groups[1]}) {groups[2]}-{groups[3]}"
                print(f"  {normalized}")
    
    def noise_and_artifacts(self):
        """Show common text noise"""
        print("\n=== TEXT NOISE AND ARTIFACTS ===")
        
        noisy_texts = [
            "This text has HTML <b>tags</b> and &amp; entities.",
            "Text with URLs: Visit https://example.com for more!",
            "Social media: Follow @username and check #hashtag",
            "Email addresses: contact@company.com and info@site.org",
            "Text with... excessive... punctuation!!!???",
            "SHOUTING TEXT WITH CAPS",
            "text with    multiple    spaces",
            "Text\nwith\nnewlines\tand\ttabs"
        ]
        
        def clean_text(text):
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            # Remove HTML entities
            text = re.sub(r'&\w+;', '', text)
            # Remove URLs
            text = re.sub(r'http\S+|www\S+', '', text)
            # Remove mentions and hashtags
            text = re.sub(r'@\w+|#\w+', '', text)
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            # Normalize punctuation
            text = re.sub(r'[.]{2,}', '.', text)
            text = re.sub(r'[!]{2,}', '!', text)
            text = re.sub(r'[?]{2,}', '?', text)
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        
        print("Noisy texts and cleaned versions:")
        for text in noisy_texts:
            cleaned = clean_text(text)
            print(f"Original: {text}")
            print(f"Cleaned:  {cleaned}\n")

challenges = TextChallenges()
challenges.encoding_issues()
challenges.inconsistent_formats()
challenges.noise_and_artifacts()
```

## 2. Basic Text Cleaning {#basic-cleaning}

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class BasicTextCleaner:
    """Comprehensive text cleaning toolkit"""
    
    def __init__(self, language='english'):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        
    def remove_special_characters(self, text, keep_spaces=True):
        """Remove special characters from text"""
        if keep_spaces:
            pattern = r'[^a-zA-Z0-9\s]'
        else:
            pattern = r'[^a-zA-Z0-9]'
        return re.sub(pattern, '', text)
    
    def remove_numbers(self, text):
        """Remove numeric characters"""
        return re.sub(r'\d+', '', text)
    
    def remove_extra_whitespace(self, text):
        """Remove extra whitespace and normalize spacing"""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def convert_case(self, text, case='lower'):
        """Convert text case"""
        if case == 'lower':
            return text.lower()
        elif case == 'upper':
            return text.upper()
        elif case == 'title':
            return text.title()
        elif case == 'sentence':
            return text.capitalize()
        return text
    
    def remove_stopwords(self, text):
        """Remove stopwords"""
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
        return ' '.join(filtered_tokens)
    
    def clean_pipeline(self, text, steps=None):
        """Apply a series of cleaning steps"""
        if steps is None:
            steps = [
                'convert_case',
                'remove_special_characters', 
                'remove_extra_whitespace',
                'remove_stopwords'
            ]
        
        cleaned_text = text
        
        for step in steps:
            if step == 'convert_case':
                cleaned_text = self.convert_case(cleaned_text)
            elif step == 'remove_special_characters':
                cleaned_text = self.remove_special_characters(cleaned_text)
            elif step == 'remove_numbers':
                cleaned_text = self.remove_numbers(cleaned_text)
            elif step == 'remove_extra_whitespace':
                cleaned_text = self.remove_extra_whitespace(cleaned_text)
            elif step == 'remove_stopwords':
                cleaned_text = self.remove_stopwords(cleaned_text)
        
        return cleaned_text
    
    def demonstrate_cleaning_effects(self, sample_texts):
        """Show effects of different cleaning steps"""
        print("=== CLEANING PIPELINE DEMONSTRATION ===")
        
        steps_combinations = [
            [],
            ['convert_case'],
            ['convert_case', 'remove_special_characters'],
            ['convert_case', 'remove_special_characters', 'remove_extra_whitespace'],
            ['convert_case', 'remove_special_characters', 'remove_extra_whitespace', 'remove_stopwords']
        ]
        
        step_names = [
            'Original',
            'Lowercase',
            '+ Remove Special Chars',
            '+ Remove Extra Spaces',
            '+ Remove Stopwords'
        ]
        
        for text in sample_texts[:2]:  # Process first 2 texts
            print(f"\nOriginal: {text}")
            print("-" * 50)
            
            for steps, name in zip(steps_combinations, step_names):
                if steps:
                    cleaned = self.clean_pipeline(text, steps)
                else:
                    cleaned = text
                print(f"{name:20}: {cleaned}")

# Demonstrate basic cleaning
cleaner = BasicTextCleaner()
sample_texts = [
    "Hello World! This is a SAMPLE text with 123 numbers and @#$% symbols.",
    "The Quick Brown Fox Jumps Over the Lazy Dog... Really?"
]

cleaner.demonstrate_cleaning_effects(sample_texts)
```

## 3. Tokenization Techniques {#tokenization}

```python
import re
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer
from nltk.tokenize import WhitespaceTokenizer, WordPunctTokenizer

class TokenizationMethods:
    """Various tokenization approaches"""
    
    def __init__(self):
        self.tweet_tokenizer = TweetTokenizer()
        self.whitespace_tokenizer = WhitespaceTokenizer()
        self.wordpunct_tokenizer = WordPunctTokenizer()
    
    def basic_tokenization(self, text):
        """Basic tokenization methods"""
        print("=== BASIC TOKENIZATION ===")
        print(f"Text: {text}\n")
        
        # Split by whitespace
        whitespace_tokens = text.split()
        print(f"Whitespace split: {whitespace_tokens}")
        
        # NLTK word tokenizer
        word_tokens = word_tokenize(text)
        print(f"NLTK word tokenize: {word_tokens}")
        
        # NLTK sentence tokenizer
        sentences = sent_tokenize(text)
        print(f"Sentence tokenize: {sentences}")
        
        return {
            'whitespace': whitespace_tokens,
            'word_tokens': word_tokens,
            'sentences': sentences
        }
    
    def regex_tokenization(self, text):
        """Regex-based tokenization"""
        print("\n=== REGEX TOKENIZATION ===")
        print(f"Text: {text}\n")
        
        # Different regex patterns
        patterns = {
            'words_only': r'\b[a-zA-Z]+\b',
            'words_and_numbers': r'\b\w+\b',
            'preserve_contractions': r"\b\w+(?:'\w+)?\b",
            'capture_punctuation': r"\w+|[.,!?;]",
            'social_media': r'@\w+|#\w+|\w+|[.,!?;]'
        }
        
        for name, pattern in patterns.items():
            tokens = re.findall(pattern, text)
            print(f"{name:20}: {tokens}")
        
        return patterns
    
    def specialized_tokenizers(self, text):
        """Specialized tokenization for different domains"""
        print("\n=== SPECIALIZED TOKENIZERS ===")
        print(f"Text: {text}\n")
        
        # Tweet tokenizer (handles social media text)
        tweet_tokens = self.tweet_tokenizer.tokenize(text)
        print(f"Tweet tokenizer: {tweet_tokens}")
        
        # Whitespace tokenizer
        whitespace_tokens = self.whitespace_tokenizer.tokenize(text)
        print(f"Whitespace tokenizer: {whitespace_tokens}")
        
        # Word + punctuation tokenizer
        wordpunct_tokens = self.wordpunct_tokenizer.tokenize(text)
        print(f"WordPunct tokenizer: {wordpunct_tokens}")
        
        return {
            'tweet': tweet_tokens,
            'whitespace': whitespace_tokens,
            'wordpunct': wordpunct_tokens
        }
    
    def custom_tokenizer(self, text, preserve_case=False, include_punctuation=False):
        """Build a custom tokenizer"""
        print("\n=== CUSTOM TOKENIZER ===")
        
        # Step 1: Handle contractions
        contractions = {
            "don't": "do not",
            "won't": "will not", 
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        expanded_text = text
        for contraction, expansion in contractions.items():
            expanded_text = re.sub(contraction, expansion, expanded_text, flags=re.IGNORECASE)
        
        # Step 2: Tokenize
        if include_punctuation:
            pattern = r"\w+|[.,!?;]"
        else:
            pattern = r"\w+"
        
        tokens = re.findall(pattern, expanded_text)
        
        # Step 3: Case handling
        if not preserve_case:
            tokens = [token.lower() for token in tokens]
        
        print(f"Original: {text}")
        print(f"Expanded: {expanded_text}")
        print(f"Tokens: {tokens}")
        
        return tokens
    
    def subword_tokenization_demo(self, text):
        """Demonstrate subword tokenization concepts"""
        print("\n=== SUBWORD TOKENIZATION CONCEPTS ===")
        
        # Simple BPE-like approach (conceptual)
        def simple_bpe(text, max_merges=5):
            """Simplified Byte Pair Encoding demonstration"""
            tokens = list(text.replace(' ', '_'))
            
            print(f"Initial tokens: {tokens}")
            
            for merge_round in range(max_merges):
                # Count pairs
                pairs = {}
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pairs[pair] = pairs.get(pair, 0) + 1
                
                if not pairs:
                    break
                
                # Find most frequent pair
                best_pair = max(pairs, key=pairs.get)
                
                # Merge the best pair
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if (i < len(tokens) - 1 and 
                        tokens[i] == best_pair[0] and 
                        tokens[i + 1] == best_pair[1]):
                        new_tokens.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                
                tokens = new_tokens
                print(f"Merge {merge_round + 1}: {tokens}")
            
            return tokens
        
        # Character-level tokenization
        char_tokens = list(text)
        print(f"Character tokens: {char_tokens}")
        
        # BPE demonstration
        bpe_tokens = simple_bpe(text)
        
        return char_tokens, bpe_tokens

# Demonstrate tokenization methods
tokenizer = TokenizationMethods()

# Test texts
test_texts = [
    "Hello world! How are you today?",
    "I can't believe it's working. Don't you think so?",
    "Check out @username and #hashtag on Twitter! 😊",
    "Visit https://example.com or email info@company.com"
]

for text in test_texts[:2]:
    tokenizer.basic_tokenization(text)
    tokenizer.regex_tokenization(text)
    tokenizer.specialized_tokenizers(text)
    tokenizer.custom_tokenizer(text, preserve_case=False, include_punctuation=True)
    tokenizer.subword_tokenization_demo(text)
    print("\n" + "="*60 + "\n")
```

## 4. Text Normalization {#normalization}

```python
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
import unicodedata

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextNormalizer:
    """Advanced text normalization techniques"""
    
    def __init__(self):
        self.porter_stemmer = PorterStemmer()
        self.snowball_stemmer = SnowballStemmer('english')
        self.lemmatizer = WordNetLemmatizer()
    
    def unicode_normalization(self, text):
        """Handle Unicode normalization"""
        print("=== UNICODE NORMALIZATION ===")
        
        # Example with accented characters
        test_text = "naïve café résumé"
        print(f"Original: {test_text}")
        
        # Different normalization forms
        nfc = unicodedata.normalize('NFC', test_text)
        nfd = unicodedata.normalize('NFD', test_text)
        nfkc = unicodedata.normalize('NFKC', test_text)
        nfkd = unicodedata.normalize('NFKD', test_text)
        
        print(f"NFC:  {nfc} (length: {len(nfc)})")
        print(f"NFD:  {nfd} (length: {len(nfd)})")
        print(f"NFKC: {nfkc} (length: {len(nfkc)})")
        print(f"NFKD: {nfkd} (length: {len(nfkd)})")
        
        # Remove diacritics
        no_diacritics = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
        print(f"No diacritics: {no_diacritics}")
        
        return nfc, nfd, nfkc, nfkd, no_diacritics
    
    def stemming_comparison(self, words):
        """Compare different stemming algorithms"""
        print("\n=== STEMMING COMPARISON ===")
        
        results = []
        
        for word in words:
            porter_stem = self.porter_stemmer.stem(word)
            snowball_stem = self.snowball_stemmer.stem(word)
            
            results.append({
                'original': word,
                'porter': porter_stem,
                'snowball': snowball_stem
            })
        
        # Display results
        print(f"{'Original':<15} {'Porter':<15} {'Snowball':<15}")
        print("-" * 45)
        for result in results:
            print(f"{result['original']:<15} {result['porter']:<15} {result['snowball']:<15}")
        
        return results
    
    def lemmatization_with_pos(self, text):
        """Demonstrate lemmatization with POS tagging"""
        print("\n=== LEMMATIZATION WITH POS TAGS ===")
        
        # Download POS tagger if needed
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        
        from nltk import pos_tag
        
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        # Map POS tags to WordNet POS tags
        def get_wordnet_pos(treebank_tag):
            if treebank_tag.startswith('J'):
                return 'a'  # adjective
            elif treebank_tag.startswith('V'):
                return 'v'  # verb
            elif treebank_tag.startswith('N'):
                return 'n'  # noun
            elif treebank_tag.startswith('R'):
                return 'r'  # adverb
            else:
                return 'n'  # default to noun
        
        lemmatized_tokens = []
        
        print(f"{'Word':<15} {'POS':<10} {'Lemma':<15}")
        print("-" * 40)
        
        for word, pos in pos_tags:
            if word.isalpha():  # Only process alphabetic tokens
                wordnet_pos = get_wordnet_pos(pos)
                lemma = self.lemmatizer.lemmatize(word.lower(), wordnet_pos)
                lemmatized_tokens.append(lemma)
                print(f"{word:<15} {pos:<10} {lemma:<15}")
        
        return lemmatized_tokens
    
    def case_normalization(self, text):
        """Different case normalization strategies"""
        print("\n=== CASE NORMALIZATION ===")
        
        strategies = {
            'lowercase': text.lower(),
            'uppercase': text.upper(),
            'title_case': text.title(),
            'sentence_case': text.capitalize(),
            'preserve_acronyms': self._preserve_acronyms_case(text)
        }
        
        print(f"Original: {text}")
        for strategy, result in strategies.items():
            print(f"{strategy:<20}: {result}")
        
        return strategies
    
    def _preserve_acronyms_case(self, text):
        """Preserve case for likely acronyms"""
        words = text.split()
        processed_words = []
        
        for word in words:
            # If word is all caps and 2-4 letters, keep uppercase
            if word.isupper() and 2 <= len(word) <= 4:
                processed_words.append(word)
            else:
                processed_words.append(word.lower())
        
        return ' '.join(processed_words)
    
    def number_normalization(self, text):
        """Normalize numbers in text"""
        print("\n=== NUMBER NORMALIZATION ===")
        
        # Define number word mappings
        number_words = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
            'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
            'eighteen': '18', 'nineteen': '19', 'twenty': '20'
        }
        
        strategies = {
            'words_to_digits': self._convert_number_words_to_digits(text, number_words),
            'remove_numbers': re.sub(r'\d+', '', text),
            'replace_with_token': re.sub(r'\d+', '<NUM>', text),
            'normalize_separators': re.sub(r'(\d+)[,.](\d+)', r'\1.\2', text)
        }
        
        print(f"Original: {text}")
        for strategy, result in strategies.items():
            print(f"{strategy:<20}: {result}")
        
        return strategies
    
    def _convert_number_words_to_digits(self, text, number_words):
        """Convert written numbers to digits"""
        words = text.split()
        converted_words = []
        
        for word in words:
            if word.lower() in number_words:
                converted_words.append(number_words[word.lower()])
            else:
                converted_words.append(word)
        
        return ' '.join(converted_words)

# Demonstrate text normalization
normalizer = TextNormalizer()

# Test Unicode normalization
normalizer.unicode_normalization("test")

# Test stemming
test_words = ['running', 'ran', 'runs', 'easily', 'fairly', 'dogs', 'cats', 'better']
normalizer.stemming_comparison(test_words)

# Test lemmatization
test_sentence = "The dogs are running faster than the cats were running yesterday."
normalizer.lemmatization_with_pos(test_sentence)

# Test case normalization
test_text = "Hello World! NASA sent a rover to MARS. The USA is proud."
normalizer.case_normalization(test_text)

# Test number normalization
number_text = "I have twenty-five dollars and 3.50 cents. Call me at five five five, one two three four."
normalizer.number_normalization(number_text)
```

## Summary

This chapter covered essential text processing and tokenization techniques:

1. **Text Processing Importance**: Understanding why raw text needs preprocessing
2. **Basic Cleaning**: Fundamental text cleaning operations
3. **Tokenization**: Various approaches to breaking text into tokens
4. **Normalization**: Advanced techniques for standardizing text

### Key Takeaways:
- Text preprocessing is crucial for NLP success
- Different tokenization methods suit different applications
- Normalization helps reduce vocabulary size and improve consistency
- The choice of preprocessing steps depends on your specific task
- Building robust preprocessing pipelines requires careful consideration of edge cases

### Best Practices:
- Always analyze your data before choosing preprocessing steps
- Consider the trade-offs between information loss and standardization
- Test different approaches and measure their impact on downstream tasks
- Document your preprocessing choices for reproducibility
- Handle edge cases and special characters appropriately

---

## Exercises

1. **Custom Tokenizer**: Build a domain-specific tokenizer for social media text
2. **Preprocessing Pipeline**: Create a configurable text preprocessing pipeline
3. **Multilingual Processing**: Extend the techniques to handle multiple languages
4. **Performance Analysis**: Compare preprocessing approaches on a real dataset
5. **Special Domains**: Adapt preprocessing for technical, medical, or legal text

---

*Effective text processing is the foundation of successful NLP applications. Master these techniques to build robust text analysis systems.* 