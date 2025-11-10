# =================================================================
# PREDICTION SCRIPT
# Author: OUSSAMA EL AZZOUZI

#
# Description:
# This script loads a pre-trained Transformer model and a data scaler
# to predict the next day's stock price based on a sample sequence.
#
# How to run:
# python predict.py
# =================================================================

import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, MultiHeadAttention, Dense, Dropout
import numpy as np
import joblib
import os

# --- Custom Transformer Encoder Layer Definition ---
# This class MUST be defined for Keras to be able to load the custom model.
# --- Define the custom layer exactly as it was during training ---
@tf.keras.utils.register_keras_serializable()
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    # +++ The build method that solves the warning +++
    def build(self, input_shape):
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ff_dim, activation="relu"),
            tf.keras.layers.Dense(self.embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)
        super(TransformerEncoder, self).build(input_shape)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output))
        return out2

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({
            "embed_dim": self.embed_dim, "num_heads": self.num_heads,
            "ff_dim": self.ff_dim, "rate": self.rate,
        })
        return config


# --- Main Prediction Logic ---
def main():
    """Main function to load artifacts and run prediction."""
    print("="*50)
    print("Stock Price Prediction Script")
    print("="*50)

    # --- Define required file paths ---
    MODEL_FILE = 'stock_prediction_transformer.keras'
    SCALER_FILE = 'data_scaler.pkl'
    TEST_DATA_FILE = 'X_test.npy'

    # --- Check if all required files exist ---
    required_files = [MODEL_FILE, SCALER_FILE, TEST_DATA_FILE]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"\n[ERROR] Missing required file: '{file_path}'")
            print("Please make sure all model artifacts are in the same directory as this script.")
            return # Exit the script

    try:
        # 1. Load the saved model
        print(f"\nLoading model from '{MODEL_FILE}'...")
        # The custom_objects argument is automatically handled by @register_keras_serializable
        loaded_model = tf.keras.models.load_model(MODEL_FILE)
        print("Model loaded successfully.")

        # 2. Load the scaler
        print(f"Loading scaler from '{SCALER_FILE}'...")
        scaler = joblib.load(SCALER_FILE)
        print("Scaler loaded successfully.")

        # 3. Load a sample from the test data
        print(f"Loading sample data from '{TEST_DATA_FILE}'...")
        X_test_sample = np.load(TEST_DATA_FILE)
        # Let's use the very last sequence from our test data as an example
        sample_sequence = X_test_sample[-1:] # Shape will be (1, 60, 7)
        print("Sample data loaded.")

        # 4. Make a prediction
        print("\n--- Making a prediction on the latest available sequence ---")
        prediction_scaled = loaded_model.predict(sample_sequence)
        print(f"Normalized (scaled) prediction: {prediction_scaled[0][0]:.6f}")

        # 5. Inverse transform the prediction to get the actual price
        # We need to create a dummy array with the same number of features as the scaler expects
        num_features = scaler.n_features_in_
        dummy_array = np.zeros((1, num_features))
        
        # The 'Close' price was the first feature (index 0) during scaling
        close_price_position = 0
        dummy_array[0, close_price_position] = prediction_scaled[0, 0]
        
        predicted_price = scaler.inverse_transform(dummy_array)[0, close_price_position]
        
        print("\n" + "="*50)
        print(f">>> Predicted Next Day's Close Price: ${predicted_price:.2f}")
        print("="*50)

    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred during prediction: {e}")

if __name__ == "__main__":
    main()

