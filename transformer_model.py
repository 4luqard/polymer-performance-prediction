import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import os


def set_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


class TransformerModel:
    """T5-style transformer encoder-decoder for supervised polymer property prediction."""
    
    def __init__(self, input_dim, target_dim=5, latent_dim=32, num_heads=4,
                 num_encoder_layers=2, num_decoder_layers=2, ff_dim=64,
                 dropout_rate=0.1, random_state=42):
        """
        Initialize T5-style transformer.
        
        Args:
            input_dim: Dimension of input SMILES features
            target_dim: Number of target properties (5)
            latent_dim: Dimension of latent representation
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder blocks
            num_decoder_layers: Number of decoder blocks
            ff_dim: Feed-forward network dimension
            dropout_rate: Dropout rate
            random_state: Random seed
        """
        self.input_dim = input_dim
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
        
        set_seeds(random_state)
    
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
        ffn_output = layers.Dense(self.ff_dim, activation="gelu")(out1)
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
        ffn_output = layers.Dense(self.ff_dim, activation="gelu")(out2)
        ffn_output = layers.Dropout(self.dropout_rate)(ffn_output)
        ffn_output = layers.Dense(self.latent_dim)(ffn_output)
        ffn_output = layers.Dropout(self.dropout_rate)(ffn_output)
        return layers.LayerNormalization(epsilon=1e-6)(out2 + ffn_output)
    
    def build_model(self):
        """Build the transformer encoder-decoder model."""
        set_seeds(self.random_state)
        
        # Encoder
        encoder_inputs = keras.Input(shape=(self.input_dim,), name="encoder_input")
        
        # Project to latent dim and reshape for transformer
        x = layers.Dense(self.latent_dim)(encoder_inputs)
        x = layers.Reshape((1, self.latent_dim))(x)
        
        # Positional encoding (simple learned embedding)
        pos_embedding = layers.Embedding(input_dim=1, output_dim=self.latent_dim)
        positions = layers.Lambda(lambda x: tf.zeros((tf.shape(x)[0], 1), dtype=tf.int32))(x)
        x = x + pos_embedding(positions)
        
        # Encoder blocks
        for _ in range(self.num_encoder_layers):
            x = self._encoder_block(x)
        
        encoder_outputs = x
        
        # Extract latent representation (flatten)
        latent_representation = layers.Flatten(name="latent")(encoder_outputs)
        
        # Build encoder model for extracting latent features
        self.encoder_model = keras.Model(encoder_inputs, latent_representation, name="encoder")
        
        # Decoder
        decoder_inputs = keras.Input(shape=(self.target_dim,), name="decoder_input")
        
        # Project decoder input
        dec = layers.Dense(self.latent_dim)(decoder_inputs)
        dec = layers.Reshape((1, self.latent_dim))(dec)
        
        # Decoder blocks with cross-attention
        for _ in range(self.num_decoder_layers):
            dec = self._decoder_block(dec, encoder_outputs)
        
        # Final projection to targets
        dec = layers.Flatten()(dec)
        dec = layers.Dense(self.ff_dim, activation="gelu")(dec)
        dec = layers.Dropout(self.dropout_rate)(dec)
        outputs = layers.Dense(self.target_dim, name="predictions")(dec)
        
        # Full model
        self.model = keras.Model(
            inputs=[encoder_inputs, decoder_inputs],
            outputs=outputs,
            name="transformer"
        )
        
        return self.model
    
    def fit(self, X, y, epochs=20, batch_size=32, validation_split=0.1, verbose=1):
        """
        Train the transformer model.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples, n_targets) with possible NaN values
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split fraction
            verbose: Verbosity level
        
        Returns:
            Training history
        """
        set_seeds(self.random_state)
        
        if self.model is None:
            self.build_model()
        
        # Handle missing values
        y_filled = np.nan_to_num(y, nan=0.0)
        sample_weights = (~np.isnan(y)).astype(np.float32).mean(axis=1)
        
        # Create decoder input (zeros for simplicity)
        decoder_input = np.zeros((X.shape[0], self.target_dim), dtype=np.float32)
        
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
            initial_learning_rate=5e-4,
            decay_steps=50,
            decay_rate=0.95
        )
        
        self.model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule),
            loss=masked_mse,
            metrics=['mae']
        )
        
        # Early stopping for convergence
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_split > 0 else 'loss',
                patience=3,
                restore_best_weights=True,
                verbose=0
            )
        ]
        
        # Train model
        history = self.model.fit(
            [X, decoder_input],
            y_filled,
            sample_weight=sample_weights,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def transform(self, X):
        """
        Extract latent features from input.
        
        Args:
            X: Input features (n_samples, n_features)
        
        Returns:
            Latent features (n_samples, latent_dim)
        """
        if self.encoder_model is None:
            raise ValueError("Model must be fitted before transform")
        
        return self.encoder_model.predict(X, verbose=0)
    
    def predict(self, X):
        """
        Predict target values.
        
        Args:
            X: Input features (n_samples, n_features)
        
        Returns:
            Predictions (n_samples, n_targets)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before predict")
        
        decoder_input = np.zeros((X.shape[0], self.target_dim), dtype=np.float32)
        return self.model.predict([X, decoder_input], verbose=0)