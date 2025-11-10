# Stock Price Prediction using a Transformer Model

This project is a deep learning-based solution for predicting the next day's closing price of a stock (using Apple Inc. - AAPL as a case study). It leverages a **Transformer Encoder** architecture, a state-of-the-art model commonly used in Natural Language Processing, adapted here for time-series forecasting.

The entire project, from data collection to model evaluation, is documented in a comprehensive Jupyter Notebook. A standalone Python script is also provided for quick, on-demand predictions.

---
## 📈 Model Performance & Results

After training the model on historical stock data from 2020 to 2025, the model was evaluated on a test set it had never seen before.

-   **Final Mean Absolute Error (MAE) on Test Data: $9.12**

This means that, on average, the model's prediction was approximately $9.12 off the actual closing price, which is a strong result for a volatile stock like AAPL.

The graph below visualizes the model's predictions (red dashed line) against the actual stock prices (blue line) on the test set.

### Actual Price vs. Predicted Price

![Stock Prediction Results](prediction_graph.png)

As shown, the model successfully captures the general trend of the stock price, demonstrating its ability to learn underlying patterns from the time-series data.

---
## 🛠️ Technologies & Libraries Used

This project was built using the following technologies:

-   **Python 3.10+**
-   **TensorFlow & Keras:** For building and training the deep learning model.
-   **Scikit-learn:** For data preprocessing (MinMaxScaler) and splitting.
-   **Pandas & Pandas-TA:** For data manipulation and calculating technical indicators (RSI, MACD).
-   **NumPy:** For numerical operations.
-   **yfinance:** For downloading historical stock market data.
-   **Matplotlib:** For plotting the results.
-   **Jupyter Notebook:** For experimentation and detailed documentation.

---
## 📂 Repository Structure

The repository is organized as follows:

```
.
├── 📄 README.md                # This file: Project explanation
├── 📓 Stock_Prediction_Transformer.ipynb  # The main Jupyter Notebook with all steps
├── 🐍 predict.py                 # Standalone script for making a single prediction
├── 🧠 stock_prediction_transformer.keras # The pre-trained Keras model
├── ⚖️ data_scaler.pkl            # The saved Scikit-learn scaler object
├── 🧪 X_test.npy                 # The test data used for prediction
├── 🎯 y_test.npy                 # The true target values for the test data
└── 📊 prediction_graph.png       # The output graph showing model performance
```

---

### Prerequisites

First, clone the repository and install the required packages. It is highly recommended to use a virtual environment.

```bash
# 1. Clone the repository (or just download the files)
git clone https://github.com/oussamaaz03/Stock-Price-Prediction-with-Transformers.git
cd Stock-Price-Prediction-with-Transformers

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# 3. Install the required libraries
pip install tensorflow scikit-learn pandas pandas_ta yfinance numpy matplotlib
```
### Option 1: Run the Standalone Prediction Script

This is the quickest way to see the model in action. This script loads the pre-trained model and makes a prediction on the last data sequence from the test set.

```bash
# Make sure you are in the project directory
python predict.py
```

You should see an output similar to this:

```
==================================================
      Stock Price Prediction Script
==================================================

... (loading messages ) ...

==================================================
>>> Predicted Next Day's Close Price: $236.38
==================================================
```

### Option 2: Explore the Jupyter Notebook

To understand the entire process, from data collection to model training and evaluation, you can run the `Stock_Prediction_Transformer.ipynb` notebook.

```bash
# 1. Install Jupyter if you haven't already
pip install jupyter

# 2. Run the Jupyter Notebook
jupyter notebook
```

This will open a new tab in your browser. Click on `Stock_Prediction_Transformer.ipynb` to open it. You can then run the cells one by one to see how the project was built.

---
## 🧠 Key Learnings & Project Insights

-   **Transformer for Time-Series:** This project demonstrates that the Transformer architecture, while famous for NLP, is highly effective for capturing long-range dependencies in time-series data.
-   **Feature Engineering:** Integrating technical indicators like **RSI** (Relative Strength Index) and **MACD** (Moving Average Convergence Divergence) provided the model with richer context, improving its predictive power.
-   **Model Persistence:** The project correctly implements saving and loading of both the Keras model (with custom layers) and the Scikit-learn scaler, which is crucial for real-world deployment.
-   **Standalone Inference:** The creation of `predict.py` separates the training environment from the prediction (inference) environment, which is a best practice in MLOps.
