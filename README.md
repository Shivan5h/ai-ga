
# AI-GA: Abstract Classification with LSTM, BERT, and Hybrid Models

This project demonstrates text classification of academic abstracts using LSTM, BERT, and a Hybrid BERT + LSTM model.

## 🔗 Colab Link
[Open in Colab](https://colab.research.google.com/github/Shivan5h/ai-ga/blob/main/ai_ga_(1).ipynb)

## 📦 Requirements
- Python
- TensorFlow
- Transformers (Hugging Face)
- Pandas, NumPy
- Matplotlib
- Scikit-learn

## 📁 Dataset
The dataset is downloaded directly from:
```
https://github.com/panagiotisanagnostou/AI-GA/blob/main/ai-ga-dataset.csv?raw=true
```

It contains:
- `abstract`: Academic abstract (text)
- `label`: Binary classification label

## 🔄 Preprocessing
- Convert to lowercase
- Remove punctuation and digits
- Remove empty or NaN abstracts
- Convert labels to integers

## 📊 Train-Test Split
- 80% training / 20% testing
- Stratified split to maintain label balance

## 📚 Models Used

### 1. LSTM Model
- Tokenized using Keras Tokenizer
- Padded sequences to max length
- Bi-directional LSTM layers with dropout
- Final output layer with sigmoid activation

### 2. BERT Model
- Uses Hugging Face BERT tokenizer and `TFBertModel`
- Inputs: `input_ids`, `attention_mask`
- Output: Pooler output → Dense → Sigmoid

### 3. Hybrid BERT + LSTM Model
- Uses `last_hidden_state` of BERT
- Passes through LSTM layer → Dense layer

## 🔍 Evaluation
Each model is evaluated on:
- Test Accuracy
- Test Loss

Best model performance is logged and compared.

## 📉 Early Stopping
- Early stopping with patience of 2 epochs to avoid overfitting.

## ⚙️ Configuration
```python
MAX_WORDS = 10000
MAX_LENGTH_LSTM = 200
MAX_LENGTH_BERT = 200
LSTM_DROPOUT_RATE = 0.3
EPOCHS = 10
BATCH_SIZE = 32
```

## 📈 Output
Prints the test accuracy and loss for:
- LSTM Model
- BERT Model
- Hybrid Model

---

*Developed using TensorFlow and Hugging Face Transformers.*
