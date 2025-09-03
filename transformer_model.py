import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from typing import Optional, Tuple, Union
from src.utils.seed import set_global_seed

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SMILESTokenizer:
    """Character-level SMILES tokenizer."""
    
    def __init__(self, max_length=150):
        self.max_length = max_length
        self.char_to_idx = {}
        self.idx_to_char = {}
        # Common SMILES characters
        self.chars = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz()[]+-=#@/*.:,;$%&\\|')
        self._build_vocab()
    
    def _build_vocab(self):
        """Build character vocabulary for fallback."""
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for i, char in enumerate(self.chars, 2):
            self.char_to_idx[char] = i
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
    
    def tokenize(self, smiles_list):
        """Convert SMILES strings to token indices."""
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        elif hasattr(smiles_list, 'values'):  # pandas Series
            smiles_list = smiles_list.values
        
        tokenized = []
        
        for smiles in smiles_list:
            tokens = []
            smiles_str = str(smiles) if smiles is not None else ''
            for char in smiles_str[:self.max_length]:
                tokens.append(self.char_to_idx.get(char, self.char_to_idx['<UNK>']))
            # Pad to max_length
            while len(tokens) < self.max_length:
                tokens.append(self.char_to_idx['<PAD>'])
            tokenized.append(tokens[:self.max_length])
        
        return np.array(tokenized, dtype=np.int32)


class TransformerModel:
    """T5-style transformer encoder-decoder for supervised polymer property prediction."""
    
    def __init__(self, vocab_size=None, target_dim=5, latent_dim=32, num_heads=4,
                 num_encoder_layers=1, num_decoder_layers=1, ff_dim=64,
                 dropout_rate=0.1, random_state=42, max_length=150):
        """
        Initialize T5-style transformer.
        
        Args:
            vocab_size: Size of vocabulary for SMILES tokenization
            target_dim: Number of target properties (5)
            latent_dim: Dimension of latent representation
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder blocks (default 1)
            num_decoder_layers: Number of decoder blocks (default 1)
            ff_dim: Feed-forward network dimension
            dropout_rate: Dropout rate
            random_state: Random seed
            max_length: Maximum SMILES string length
        """
        self.tokenizer = SMILESTokenizer(max_length=max_length)
        self.vocab_size = vocab_size or self.tokenizer.vocab_size
        self.max_length = max_length
        self.target_dim = target_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        
        self.model = None
        self.encoder_model = None
        
        set_global_seed(random_state)
    
    def _encoder_block(self, inputs):
        """Single transformer encoder block."""
        # Multi-head self-attention
        attn_output = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.latent_dim // self.num_heads,
            dropout=self.dropout_rate
        )(inputs, inputs)
        attn_output = layers.Dropout(self.dropout_rate)(attn_output)
        out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = layers.Dense(self.ff_dim, activation="linear")(out1)
        ffn_output = layers.Dropout(self.dropout_rate)(ffn_output)
        ffn_output = layers.Dense(self.latent_dim)(ffn_output)
        ffn_output = layers.Dropout(self.dropout_rate)(ffn_output)
        return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    
    def _decoder_block(self, dec_inputs, enc_outputs):
        """Single transformer decoder block with cross-attention."""
        # Self-attention
        attn1 = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.latent_dim // self.num_heads,
            dropout=self.dropout_rate
        )(dec_inputs, dec_inputs)
        attn1 = layers.Dropout(self.dropout_rate)(attn1)
        out1 = layers.LayerNormalization(epsilon=1e-6)(dec_inputs + attn1)
        
        # Cross-attention with encoder outputs
        attn2 = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.latent_dim // self.num_heads,
            dropout=self.dropout_rate
        )(out1, enc_outputs, enc_outputs)
        attn2 = layers.Dropout(self.dropout_rate)(attn2)
        out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + attn2)
        
        # Feed-forward
        ffn_output = layers.Dense(self.ff_dim, activation="linear")(out2)
        ffn_output = layers.Dropout(self.dropout_rate)(ffn_output)
        ffn_output = layers.Dense(self.latent_dim)(ffn_output)
        ffn_output = layers.Dropout(self.dropout_rate)(ffn_output)
        return layers.LayerNormalization(epsilon=1e-6)(out2 + ffn_output)
    
    def build_model(self):
        """Build the transformer encoder-decoder model."""
        set_global_seed(self.random_state)
        
        # Input: tokenized SMILES sequences
        encoder_inputs = keras.Input(shape=(self.max_length,), dtype=tf.int32, name='smiles_tokens')
        
        # Embedding layer for tokens
        embedded = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.latent_dim,
            mask_zero=True  # Mask padding tokens
        )(encoder_inputs)
        
        # Positional encoding
        positions = layers.Lambda(
            lambda x: tf.tile(
                tf.expand_dims(tf.range(start=0, limit=self.max_length, delta=1), 0),
                [tf.shape(x)[0], 1]
            )
        )(encoder_inputs)
        
        position_embeddings = layers.Embedding(
            input_dim=self.max_length,
            output_dim=self.latent_dim
        )(positions)
        
        # Add positional encoding to embeddings
        x = layers.Add()([embedded, position_embeddings])
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Encoder blocks
        for _ in range(self.num_encoder_layers):
            x = self._encoder_block(x)
        
        encoder_outputs = x
        
        # Global average pooling to get fixed-size latent representation
        latent_representation = layers.GlobalAveragePooling1D(name="latent")(encoder_outputs)
        
        # Build encoder model for extracting latent features
        self.encoder_model = keras.Model(encoder_inputs, latent_representation, name="encoder")
        
        # Decoder
        # For simplicity, use a learned decoder query
        # Create a learnable decoder query vector
        decoder_query_input = layers.Lambda(
            lambda x: tf.ones((tf.shape(x)[0], 1, 1)),
            output_shape=(1, 1)
        )(encoder_inputs)
        decoder_query = layers.Dense(self.latent_dim, name='decoder_query')(decoder_query_input)
        
        # Decoder blocks with cross-attention
        dec = decoder_query
        for _ in range(self.num_decoder_layers):
            dec = self._decoder_block(dec, encoder_outputs)
        
        # Final projection to targets
        dec = layers.Flatten()(dec)
        dec = layers.Dense(self.ff_dim, activation="linear")(dec)
        dec = layers.Dropout(self.dropout_rate)(dec)
        outputs = layers.Dense(self.target_dim, name="predictions")(dec)
        
        # Full model
        self.model = keras.Model(
            inputs=encoder_inputs,
            outputs=outputs,
            name="transformer"
        )
        
        return self.model
    
    def fit(self, X_smiles, y, epochs=10, batch_size=32, validation_split=0.1, verbose=1):
        """
        Fit the transformer model.
        
        Args:
            X_smiles: SMILES strings (n_samples,) or pandas Series
            y: Target values (n_samples, n_targets) with possible NaN values
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split fraction
            verbose: Verbosity level
        
        Returns:
            Training history
        """
        set_global_seed(self.random_state)
        
        if self.model is None:
            self.build_model()
        
        # Tokenize SMILES strings
        X_tokens = self.tokenizer.tokenize(X_smiles)
        
        # Handle missing values
        y_filled = np.nan_to_num(y, nan=0.0)
        sample_weights = (~np.isnan(y)).astype(np.float32).mean(axis=1)
        
        # Custom loss that handles missing values
        def masked_mse(y_true, y_pred):
            # Create mask for non-zero (non-missing) values
            mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
            squared_diff = tf.square(y_true - y_pred)
            masked_squared_diff = squared_diff * mask
            # Average only over non-masked values
            return tf.reduce_sum(masked_squared_diff) / (tf.reduce_sum(mask) + 1e-7)
        
        # Learning rate schedule for faster convergence
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=50,
            decay_rate=0.9
        )
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-3),  # Fixed LR for faster convergence
            loss=masked_mse,
            metrics=['mae'],
            jit_compile=False  # Disable XLA compilation for CPU
        )
        
        # Store y for potential residual calculation
        self.last_y_train = y
        
        # Early stopping for convergence
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_split > 0 else 'loss',
                patience=5,
                restore_best_weights=True,
                verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_split > 0 else 'loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=0
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_tokens,
            y_filled,
            sample_weight=sample_weights,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Output final epoch losses
        if history.history:
            final_train_loss = history.history['loss'][-1] if 'loss' in history.history else None
            final_val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else None
            print(f"\nFinal Epoch - Training Loss: {final_train_loss:.6f}, Validation Loss: {final_val_loss:.6f}")
        
        return history
    
    def fit_predict(self, X_smiles, y, epochs=10, batch_size=32, 
                    validation_split=0.1, verbose=1, return_residuals=False):
        """
        Fit the model and return predictions (and optionally residuals).
        
        Args:
            X_smiles: SMILES strings
            y: Target values
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split fraction
            verbose: Verbosity level
            return_residuals: If True, return residuals too
        
        Returns:
            Predictions or tuple of (predictions, residuals)
        """
        # Fit the model
        history = self.fit(X_smiles, y, epochs, batch_size, validation_split, verbose)
        
        # Get predictions
        predictions = self.predict(X_smiles)
        
        if return_residuals:
            residuals = y - predictions
            return predictions, residuals, history
        
        return predictions, history
    
    def get_training_residuals(self, X_smiles):
        """
        Get residuals for the training data.
        
        Args:
            X_smiles: Training SMILES strings
        
        Returns:
            Residuals (n_samples, n_targets)
        """
        if not hasattr(self, 'last_y_train') or self.last_y_train is None:
            raise ValueError("Model must be fitted before getting training residuals")
        
        predictions = self.predict(X_smiles)
        residuals = self.last_y_train - predictions
        return residuals
    
    def transform(self, X_smiles):
        """
        Extract latent features from SMILES strings.
        
        Args:
            X_smiles: SMILES strings (n_samples,) or pandas Series
        
        Returns:
            Latent features (n_samples, latent_dim)
        """
        if self.encoder_model is None:
            raise ValueError("Model must be fitted before transform")
        
        # Tokenize SMILES strings
        X_tokens = self.tokenizer.tokenize(X_smiles)
        
        return self.encoder_model.predict(X_tokens, verbose=0)
    
    def predict(self, X_smiles, return_residuals=False, y_true=None):
        """
        Predict target values from SMILES strings.
        
        Args:
            X_smiles: SMILES strings (n_samples,) or pandas Series
            return_residuals: If True and y_true provided, return residuals too
            y_true: True values for residual calculation
        
        Returns:
            Predictions (n_samples, n_targets) or tuple of (predictions, residuals)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before predict")
        
        # Tokenize SMILES strings
        X_tokens = self.tokenizer.tokenize(X_smiles)
        
        predictions = self.model.predict(X_tokens, verbose=0)
        
        if return_residuals and y_true is not None:
            residuals = y_true - predictions
            return predictions, residuals
        
        return predictions
