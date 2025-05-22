import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification

# Constants
MAX_LENGTH = 128    # Max sequence length for BERT
BATCH_SIZE = 8     # Adjust based on GPU memory
EPOCHS = 3          # Number of fine-tuning epochs
MODEL_NAME = "bert-base-uncased"  # Pre-trained BERT model

def load_and_prepare_data():
    """Load IMDB dataset via TFDS and split into train/val/test."""
    ds, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

    # Load full train split
    train_ds = ds['train'].batch(info.splits['train'].num_examples)
    train_examples, train_labels = tfds.as_numpy(next(iter(train_ds)))

    # Load full test split
    test_ds = ds['test'].batch(info.splits['test'].num_examples)
    test_examples, test_labels = tfds.as_numpy(next(iter(test_ds)))

    # Decode bytes to strings
    train_texts = [ex.decode('utf-8') for ex in train_examples]
    test_texts  = [ex.decode('utf-8') for ex in test_examples]

    # Create an 80/20 train/validation split
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=42
    )

    return (tr_texts, tr_labels), (val_texts, val_labels), (test_texts, test_labels)

def tokenize_data(texts, tokenizer):
    """Tokenize list of texts for BERT input."""
    return tokenizer(
        texts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )

def create_tf_dataset(encodings, labels):
    """Convert encodings and labels to a batched tf.data.Dataset."""
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
    return dataset.shuffle(1000).batch(BATCH_SIZE)

def build_model():
    """Instantiate and compile the BERT classification model."""
    model = TFBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # Use legacy Adam optimizer to ensure compatibility and avoid errors
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=2e-5)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def main():
    # Load & split data
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = load_and_prepare_data()

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize datasets
    train_enc = tokenize_data(train_texts, tokenizer)
    val_enc   = tokenize_data(val_texts, tokenizer)
    test_enc  = tokenize_data(test_texts, tokenizer)

    # Build tf.data.Datasets
    train_ds = create_tf_dataset(train_enc, train_labels)
    val_ds   = create_tf_dataset(val_enc,   val_labels)
    test_ds  = create_tf_dataset(test_enc,  test_labels)

    # Instantiate and train model
    model = build_model()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )

    # Plot training curves
    for metric in ['accuracy', 'loss']:
        plt.figure()
        plt.plot(history.history[metric], label=f'train_{metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'val_{metric}')
        plt.title(metric)
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(f'{metric}_plot.png')

    # Evaluate on test set
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Confusion matrix & classification report
    y_true, y_pred = [], []
    for batch in test_ds:
        inputs, labels = batch
        logits = model.predict(inputs).logits
        preds = np.argmax(logits, axis=1)
        y_pred.extend(preds)
        y_true.extend(labels.numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    print('Classification Report:')
    print(classification_report(y_true, y_pred, target_names=['negative','positive']))

    # Save model and tokenizer
    model.save_pretrained('sentiment_bert_model')
    tokenizer.save_pretrained('sentiment_bert_model')
    print("Model and tokenizer saved to 'sentiment_bert_model'.")

if __name__ == '__main__':
    main()
