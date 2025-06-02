# Chapter 11: Speech Recognition with AI

## Learning Objectives
By the end of this chapter, students will be able to:
- Understand the fundamentals of speech and audio processing
- Implement speech recognition systems using AI techniques
- Apply feature extraction methods for audio signals
- Build end-to-end speech recognition applications
- Evaluate and optimize speech recognition models

## Table of Contents
1. [Introduction to Speech Recognition](#introduction)
2. [Audio Signal Processing Fundamentals](#audio-fundamentals)
3. [Feature Extraction for Speech](#feature-extraction)
4. [Traditional Speech Recognition Methods](#traditional-methods)
5. [Deep Learning for Speech Recognition](#deep-learning)
6. [Attention Mechanisms in Speech](#attention)
7. [Modern Architectures](#modern-architectures)
8. [Implementation Examples](#implementation)
9. [Evaluation and Optimization](#evaluation)
10. [Real-world Applications](#applications)

## 1. Introduction to Speech Recognition {#introduction}

Speech recognition, also known as Automatic Speech Recognition (ASR), is the technology that converts spoken language into text. It's a fundamental component of many AI applications including virtual assistants, transcription services, and voice-controlled systems.

### Key Challenges in Speech Recognition:

1. **Acoustic Variability**: Different speakers, accents, speaking styles
2. **Environmental Noise**: Background sounds, echo, interference
3. **Linguistic Diversity**: Different languages, dialects, vocabularies
4. **Temporal Dynamics**: Variable speaking speeds and pauses
5. **Homophones**: Words that sound the same but have different meanings

### Evolution of Speech Recognition:

- **1950s**: Early experiments with single-speaker, limited vocabulary
- **1970s**: Statistical methods and Hidden Markov Models (HMMs)
- **1990s**: Large vocabulary continuous speech recognition
- **2000s**: Machine learning approaches
- **2010s**: Deep learning revolution
- **Present**: Transformer-based models and end-to-end systems

## 2. Audio Signal Processing Fundamentals {#audio-fundamentals}

### Digital Audio Representation:

```python
import numpy as np
import librosa
import matplotlib.pyplot as plt

def load_and_analyze_audio(file_path):
    """Load and analyze audio file"""
    # Load audio file
    audio, sample_rate = librosa.load(file_path, sr=16000)
    
    # Basic properties
    duration = len(audio) / sample_rate
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Number of samples: {len(audio)}")
    
    return audio, sample_rate

# Visualize audio waveform
def plot_waveform(audio, sample_rate):
    """Plot audio waveform"""
    time = np.linspace(0, len(audio) / sample_rate, len(audio))
    plt.figure(figsize=(12, 4))
    plt.plot(time, audio)
    plt.title('Audio Waveform')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
```

### Frequency Domain Analysis:

```python
def compute_spectrogram(audio, sample_rate, n_fft=2048, hop_length=512):
    """Compute and visualize spectrogram"""
    # Compute Short-Time Fourier Transform (STFT)
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    
    # Convert to dB scale
    db_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # Visualize
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(
        db_magnitude, 
        sr=sample_rate, 
        hop_length=hop_length,
        x_axis='time', 
        y_axis='hz'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.show()
    
    return db_magnitude
```

## 3. Feature Extraction for Speech {#feature-extraction}

### Mel-Frequency Cepstral Coefficients (MFCCs):

MFCCs are the most widely used features in speech recognition, capturing the spectral characteristics of speech in a compact form.

```python
def extract_mfcc_features(audio, sample_rate, n_mfcc=13):
    """Extract MFCC features from audio"""
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=2048,
        hop_length=512
    )
    
    # Add delta and delta-delta features
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    # Combine features
    features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
    
    return features

# Visualize MFCC features
def plot_mfcc(mfccs, sample_rate):
    """Plot MFCC features"""
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(mfccs, x_axis='time', sr=sample_rate)
    plt.colorbar()
    plt.title('MFCC Features')
    plt.xlabel('Time (seconds)')
    plt.ylabel('MFCC Coefficients')
    plt.show()
```

### Mel-Spectrograms:

```python
def extract_mel_spectrogram(audio, sample_rate, n_mels=80):
    """Extract mel-spectrogram features"""
    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=2048,
        hop_length=512
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

def plot_mel_spectrogram(mel_spec, sample_rate):
    """Plot mel-spectrogram"""
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(
        mel_spec, 
        x_axis='time', 
        y_axis='mel',
        sr=sample_rate
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mel Frequency')
    plt.show()
```

### Log-Mel Filter Banks:

```python
def extract_log_mel_filterbank(audio, sample_rate, n_filters=40):
    """Extract log mel filter bank features"""
    # Compute mel filter bank
    mel_filterbank = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_filters,
        n_fft=2048,
        hop_length=512,
        power=1.0  # Use magnitude instead of power
    )
    
    # Apply log compression
    log_mel_filterbank = np.log(mel_filterbank + 1e-8)
    
    return log_mel_filterbank
```

## 4. Traditional Speech Recognition Methods {#traditional-methods}

### Hidden Markov Models (HMMs):

HMMs model the temporal structure of speech by representing phonemes as states with transition probabilities.

```python
from hmmlearn import hmm
import numpy as np

class PhonemeHMM:
    def __init__(self, n_states=3, n_features=13):
        """Initialize phoneme HMM"""
        self.n_states = n_states
        self.n_features = n_features
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full"
        )
    
    def train(self, features_list, lengths):
        """Train HMM on feature sequences"""
        # Concatenate all feature sequences
        all_features = np.vstack(features_list)
        
        # Train the model
        self.model.fit(all_features, lengths)
    
    def score(self, features):
        """Compute log-likelihood of features"""
        return self.model.score(features)
    
    def decode(self, features):
        """Decode most likely state sequence"""
        return self.model.decode(features)

# Example usage
def train_phoneme_models(phoneme_data):
    """Train HMM models for each phoneme"""
    models = {}
    
    for phoneme, feature_sequences in phoneme_data.items():
        print(f"Training model for phoneme: {phoneme}")
        
        # Extract features and lengths
        features_list = []
        lengths = []
        
        for sequence in feature_sequences:
            features_list.append(sequence)
            lengths.append(len(sequence))
        
        # Train HMM
        model = PhonemeHMM()
        model.train(features_list, lengths)
        models[phoneme] = model
    
    return models
```

### Gaussian Mixture Models (GMMs):

```python
from sklearn.mixture import GaussianMixture

class GMMClassifier:
    def __init__(self, n_components=8):
        """Initialize GMM classifier"""
        self.n_components = n_components
        self.models = {}
    
    def train(self, features_dict):
        """Train GMM for each class"""
        for class_name, features in features_dict.items():
            # Flatten feature sequences
            all_features = np.vstack(features)
            
            # Train GMM
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type='full'
            )
            gmm.fit(all_features)
            
            self.models[class_name] = gmm
    
    def predict(self, features):
        """Predict class for given features"""
        scores = {}
        
        for class_name, model in self.models.items():
            scores[class_name] = model.score(features)
        
        # Return class with highest score
        return max(scores, key=scores.get), scores
```

## 5. Deep Learning for Speech Recognition {#deep-learning}

### Recurrent Neural Networks for Speech:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

class SpeechRNN:
    def __init__(self, input_dim, num_classes, hidden_units=128):
        """Initialize RNN for speech recognition"""
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.model = self._build_model()
    
    def _build_model(self):
        """Build RNN model"""
        model = models.Sequential([
            layers.LSTM(
                self.hidden_units, 
                return_sequences=True, 
                input_shape=(None, self.input_dim)
            ),
            layers.Dropout(0.2),
            layers.LSTM(self.hidden_units, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(self.hidden_units),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, x_train, y_train, x_val, y_val, epochs=50):
        """Train the model"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
```

### Connectionist Temporal Classification (CTC):

CTC enables training speech recognition models without requiring aligned transcriptions.

```python
def ctc_loss_function(y_true, y_pred):
    """CTC loss function"""
    # Get batch size and sequence lengths
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

class CTCSpeechModel:
    def __init__(self, input_dim, num_classes):
        """Initialize CTC-based speech model"""
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self):
        """Build CTC model"""
        # Input layer
        inputs = layers.Input(shape=(None, self.input_dim))
        
        # Convolutional layers for feature extraction
        x = layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        
        # Bidirectional LSTM layers
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model
    
    def compile_model(self):
        """Compile model with CTC loss"""
        self.model.compile(
            optimizer='adam',
            loss=ctc_loss_function
        )
```

## 6. Attention Mechanisms in Speech {#attention}

### Attention-based Encoder-Decoder:

```python
class AttentionLayer(layers.Layer):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)
    
    def call(self, query, values):
        # Add time axis to query
        query_with_time_axis = tf.expand_dims(query, 1)
        
        # Calculate attention scores
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)
        ))
        
        # Calculate attention weights
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Calculate context vector
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights

class SpeechAttentionModel:
    def __init__(self, vocab_size, embedding_dim, enc_units, dec_units, attention_units):
        """Initialize attention-based speech model"""
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.enc_units = enc_units
        self.dec_units = dec_units
        self.attention_units = attention_units
        
        # Build encoder and decoder
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.attention = AttentionLayer(attention_units)
    
    def _build_encoder(self):
        """Build encoder network"""
        encoder = models.Sequential([
            layers.Bidirectional(layers.LSTM(self.enc_units, return_sequences=True)),
            layers.Dropout(0.2)
        ])
        return encoder
    
    def _build_decoder(self):
        """Build decoder network"""
        decoder = models.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim),
            layers.LSTM(self.dec_units, return_state=True),
            layers.Dense(self.vocab_size)
        ])
        return decoder
```

## 7. Modern Architectures {#modern-architectures}

### Transformer for Speech Recognition:

```python
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        
        return output, attention_weights

class SpeechTransformer:
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size):
        """Initialize Speech Transformer"""
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.final_layer = layers.Dense(target_vocab_size)
    
    def _build_encoder(self):
        """Build transformer encoder"""
        encoder_layers = []
        
        for _ in range(self.num_layers):
            encoder_layer = tf.keras.Sequential([
                MultiHeadAttention(self.d_model, self.num_heads),
                layers.Dropout(0.1),
                layers.LayerNormalization(epsilon=1e-6),
                layers.Dense(self.dff, activation='relu'),
                layers.Dense(self.d_model),
                layers.Dropout(0.1),
                layers.LayerNormalization(epsilon=1e-6)
            ])
            encoder_layers.append(encoder_layer)
        
        return encoder_layers
```

### Wav2Vec 2.0 Architecture:

```python
class Wav2Vec2Model:
    def __init__(self, hidden_size=768, num_attention_heads=12, num_hidden_layers=12):
        """Initialize Wav2Vec 2.0 inspired model"""
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        
        self.feature_extractor = self._build_feature_extractor()
        self.feature_projection = self._build_feature_projection()
        self.encoder = self._build_encoder()
    
    def _build_feature_extractor(self):
        """Build convolutional feature extractor"""
        return models.Sequential([
            layers.Conv1D(512, 10, strides=5, activation='gelu'),
            layers.LayerNormalization(),
            layers.Conv1D(512, 3, strides=2, activation='gelu'),
            layers.LayerNormalization(),
            layers.Conv1D(512, 3, strides=2, activation='gelu'),
            layers.LayerNormalization(),
            layers.Conv1D(512, 3, strides=2, activation='gelu'),
            layers.LayerNormalization(),
            layers.Conv1D(512, 3, strides=2, activation='gelu'),
            layers.LayerNormalization(),
            layers.Conv1D(512, 2, strides=2, activation='gelu'),
            layers.LayerNormalization(),
            layers.Conv1D(512, 2, strides=2, activation='gelu'),
            layers.LayerNormalization()
        ])
    
    def _build_feature_projection(self):
        """Build feature projection layer"""
        return models.Sequential([
            layers.Dense(self.hidden_size),
            layers.LayerNormalization(),
            layers.Dropout(0.1)
        ])
    
    def _build_encoder(self):
        """Build transformer encoder"""
        encoder_layers = []
        
        for _ in range(self.num_hidden_layers):
            layer = models.Sequential([
                MultiHeadAttention(self.hidden_size, self.num_attention_heads),
                layers.Dropout(0.1),
                layers.LayerNormalization(),
                layers.Dense(self.hidden_size * 4, activation='gelu'),
                layers.Dense(self.hidden_size),
                layers.Dropout(0.1),
                layers.LayerNormalization()
            ])
            encoder_layers.append(layer)
        
        return encoder_layers
```

## 8. Implementation Examples {#implementation}

### Complete Speech Recognition Pipeline:

```python
import tensorflow as tf
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder

class SpeechRecognitionPipeline:
    def __init__(self, sample_rate=16000, n_mfcc=13):
        """Initialize speech recognition pipeline"""
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.label_encoder = LabelEncoder()
        self.model = None
    
    def preprocess_audio(self, audio_file):
        """Preprocess audio file"""
        # Load audio
        audio, sr = librosa.load(audio_file, sr=self.sample_rate)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=2048,
            hop_length=512
        )
        
        # Add delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Combine features
        features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
        
        return features.T  # Transpose to (time, features)
    
    def build_model(self, input_shape, num_classes):
        """Build speech recognition model"""
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(128),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, audio_files, labels, validation_split=0.2):
        """Train the speech recognition model"""
        # Extract features from all audio files
        features_list = []
        for audio_file in audio_files:
            features = self.preprocess_audio(audio_file)
            features_list.append(features)
        
        # Pad sequences to same length
        max_length = max(len(features) for features in features_list)
        padded_features = tf.keras.preprocessing.sequence.pad_sequences(
            features_list, maxlen=max_length, dtype='float32', padding='post'
        )
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Build model
        input_shape = (max_length, features_list[0].shape[1])
        num_classes = len(np.unique(encoded_labels))
        self.build_model(input_shape, num_classes)
        
        # Train model
        history = self.model.fit(
            padded_features, encoded_labels,
            validation_split=validation_split,
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        return history
    
    def predict(self, audio_file):
        """Predict speech content"""
        # Preprocess audio
        features = self.preprocess_audio(audio_file)
        
        # Pad to match training data
        max_length = self.model.input_shape[1]
        if len(features) < max_length:
            features = np.pad(features, ((0, max_length - len(features)), (0, 0)))
        else:
            features = features[:max_length]
        
        # Make prediction
        features = np.expand_dims(features, axis=0)
        prediction = self.model.predict(features)
        predicted_class = np.argmax(prediction)
        
        # Decode label
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
        confidence = np.max(prediction)
        
        return predicted_label, confidence

# Usage example
def train_speech_recognizer():
    """Example usage of speech recognition pipeline"""
    # Initialize pipeline
    pipeline = SpeechRecognitionPipeline()
    
    # Example data (replace with actual paths)
    audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
    labels = ['hello', 'goodbye', 'thank you']
    
    # Train the model
    history = pipeline.train(audio_files, labels)
    
    # Make predictions
    prediction, confidence = pipeline.predict('test_audio.wav')
    print(f"Predicted: {prediction} (confidence: {confidence:.3f})")
    
    return pipeline
```

### Real-time Speech Recognition:

```python
import pyaudio
import threading
import queue

class RealTimeSpeechRecognizer:
    def __init__(self, model, label_encoder, sample_rate=16000, chunk_size=1024):
        """Initialize real-time speech recognizer"""
        self.model = model
        self.label_encoder = label_encoder
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.is_listening = False
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback function"""
        if self.is_listening:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_data)
        return (None, pyaudio.paContinue)
    
    def start_listening(self):
        """Start listening for audio"""
        self.is_listening = True
        
        # Open audio stream
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        
        self.stream.start_stream()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.start()
    
    def stop_listening(self):
        """Stop listening for audio"""
        self.is_listening = False
        
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
    
    def process_audio(self):
        """Process audio chunks"""
        audio_buffer = []
        
        while self.is_listening:
            try:
                # Get audio chunk
                chunk = self.audio_queue.get(timeout=1)
                audio_buffer.extend(chunk)
                
                # Process when buffer is large enough
                if len(audio_buffer) >= self.sample_rate * 2:  # 2 seconds
                    audio_array = np.array(audio_buffer)
                    
                    # Extract features and predict
                    prediction = self.recognize_speech(audio_array)
                    if prediction:
                        print(f"Recognized: {prediction}")
                    
                    # Reset buffer
                    audio_buffer = []
                    
            except queue.Empty:
                continue
    
    def recognize_speech(self, audio):
        """Recognize speech from audio array"""
        try:
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=13
            )
            
            # Add delta features
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            
            # Combine features
            features = np.vstack([mfccs, delta_mfccs, delta2_mfccs]).T
            
            # Pad/truncate to model input size
            max_length = self.model.input_shape[1]
            if len(features) < max_length:
                features = np.pad(features, ((0, max_length - len(features)), (0, 0)))
            else:
                features = features[:max_length]
            
            # Make prediction
            features = np.expand_dims(features, axis=0)
            prediction = self.model.predict(features, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Return result if confidence is high enough
            if confidence > 0.7:  # Threshold
                return self.label_encoder.inverse_transform([predicted_class])[0]
            
        except Exception as e:
            print(f"Error in recognition: {e}")
        
        return None
```

## 9. Evaluation and Optimization {#evaluation}

### Evaluation Metrics:

```python
def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate (WER)"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Dynamic programming for edit distance
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(
                    d[i-1][j] + 1,      # deletion
                    d[i][j-1] + 1,      # insertion
                    d[i-1][j-1] + 1     # substitution
                )
    
    wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)
    return wer

def evaluate_speech_model(model, test_data, test_labels):
    """Evaluate speech recognition model"""
    predictions = model.predict(test_data)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_classes == test_labels)
    
    # Calculate per-class metrics
    from sklearn.metrics import classification_report
    report = classification_report(test_labels, predicted_classes)
    
    return accuracy, report
```

### Model Optimization:

```python
def optimize_speech_model(model, x_train, y_train, x_val, y_val):
    """Optimize speech recognition model"""
    # Learning rate scheduling
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
    
    # Model checkpointing
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_speech_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Train with optimization
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[lr_scheduler, early_stopping, checkpoint],
        verbose=1
    )
    
    return history
```

## 10. Real-world Applications {#applications}

### Voice Assistants:

```python
class VoiceAssistant:
    def __init__(self, speech_model, nlp_processor):
        """Initialize voice assistant"""
        self.speech_model = speech_model
        self.nlp_processor = nlp_processor
        self.recognizer = RealTimeSpeechRecognizer(speech_model, None)
    
    def process_voice_command(self, audio):
        """Process voice command"""
        # Recognize speech
        text = self.recognizer.recognize_speech(audio)
        
        if text:
            # Process natural language
            intent, entities = self.nlp_processor.process(text)
            
            # Execute command
            response = self.execute_command(intent, entities)
            
            return response
        
        return "Sorry, I didn't understand that."
    
    def execute_command(self, intent, entities):
        """Execute recognized command"""
        if intent == "weather":
            return f"The weather in {entities['location']} is sunny."
        elif intent == "time":
            return f"The current time is {datetime.now().strftime('%H:%M')}."
        else:
            return "I can help with weather and time information."
```

### Transcription Services:

```python
class TranscriptionService:
    def __init__(self, speech_model):
        """Initialize transcription service"""
        self.speech_model = speech_model
    
    def transcribe_file(self, audio_file, chunk_duration=30):
        """Transcribe audio file"""
        # Load audio
        audio, sr = librosa.load(audio_file)
        
        # Split into chunks
        chunk_samples = chunk_duration * sr
        chunks = [audio[i:i+chunk_samples] for i in range(0, len(audio), chunk_samples)]
        
        # Transcribe each chunk
        transcription = []
        for chunk in chunks:
            if len(chunk) > 0:
                text = self.transcribe_chunk(chunk, sr)
                if text:
                    transcription.append(text)
        
        return " ".join(transcription)
    
    def transcribe_chunk(self, audio_chunk, sample_rate):
        """Transcribe audio chunk"""
        # Extract features
        features = self.extract_features(audio_chunk, sample_rate)
        
        # Make prediction
        prediction = self.speech_model.predict(features)
        
        # Decode prediction to text
        text = self.decode_prediction(prediction)
        
        return text
```

## Summary

This chapter provided a comprehensive overview of speech recognition with AI, covering:

1. **Fundamentals**: Audio signal processing and feature extraction
2. **Traditional Methods**: HMMs and GMMs for speech recognition
3. **Deep Learning**: RNNs, CNNs, and attention mechanisms
4. **Modern Architectures**: Transformers and Wav2Vec 2.0
5. **Implementation**: Complete pipelines and real-time systems
6. **Applications**: Voice assistants and transcription services

### Key Takeaways:
- Feature extraction is crucial for speech recognition performance
- Deep learning models significantly outperform traditional methods
- Attention mechanisms improve handling of long sequences
- Real-time processing requires careful optimization
- Evaluation metrics like WER are essential for measuring performance

---

## Exercises

1. Implement a phoneme recognition system using HMMs
2. Build a CNN-based speech command classifier
3. Create a real-time voice activity detector
4. Compare MFCC vs. mel-spectrogram features
5. Develop a multilingual speech recognition system

---

*Master speech recognition through hands-on implementation of various techniques and architectures.* 