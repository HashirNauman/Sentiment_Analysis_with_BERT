# Sentiment Analysis with BERT on IMDB Reviews

A concise implementation of a binary sentiment classifier using Hugging Face's BERT (`bert-base-uncased`) fine-tuned on the IMDB movie reviews dataset via TensorFlow.

---

## 🚀 Features

* **Dataset**: IMDB reviews (binary classification: positive vs. negative)
* **Model**: `TFBertForSequenceClassification` (`bert-base-uncased`)
* **Training**: 3 epochs, batch size 8, learning rate 2e-5
* **Evaluation**: Test accuracy \~88%, confusion matrix, and classification report
* **Outputs**: Saved model/tokenizer, training curves, confusion matrix plot

---

## 📦 Requirements

* Python 3.9+
* `tensorflow` (>=2.10)
* `tensorflow_datasets`
* `transformers`
* `scikit-learn`
* `matplotlib`
* `seaborn`

Install via:

```bash
pip install tensorflow tensorflow_datasets transformers scikit-learn matplotlib seaborn
```

---

## 📝 Usage

1. **Clone the repo**

   
   

git clone <your-repo-url>
cd <repo-folder>

````

2. **Run training & evaluation**
   ```bash
python main.py
````

* Trains for 3 epochs
* Saves `accuracy_plot.png`, `loss_plot.png`, and `confusion_matrix.png`
* Prints final test loss and accuracy (\~88%)

3. **Result artifacts**

   * `sentiment_bert_model/`

     * `tf_model.h5` and tokenizer files
   * Plot images in the working directory

---

## 📊 Evaluation

* **Test Accuracy**: \~0.88

* **Confusion Matrix**:

  * True Negative: 10,900
  * False Positive: 1,600
  * False Negative: 1,399
  * True Positive: 11,101

* **Classification Report**:

  * Precision/Recall/F1 for both classes

---

## 📂 File Structure

```
├── Main.py              # Training & evaluation script
├── accuracy_plot.png    # Training/validation accuracy over epochs
├── loss_plot.png        # Training/validation loss over epochs
├── confusion_matrix.png # Confusion matrix heatmap
└── sentiment_bert_model # Saved model & tokenizer
```

---

## 🔧 Customization

* **Hyperparameters**: Modify `MAX_LENGTH`, `BATCH_SIZE`, `EPOCHS`, and learning rate in `main.py`.
* **Model**: Swap `MODEL_NAME` for other BERT variants (e.g., `bert-large-uncased`).

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*Happy coding!*
