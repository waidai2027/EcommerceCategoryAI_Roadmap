# train.py

import pickle
import logging
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from data_preprocessing import load_data, preprocess_data

# Set up logging
logging.basicConfig(
    filename='training_log.log',  # Log file name
    level=logging.DEBUG,           # Set the logging level to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
)

def train_model(X_train, y_train):
    """Train the Naive Bayes model."""
    logging.info("Starting model training...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    logging.info("Model training completed.")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print the classification report."""
    logging.info("Starting model evaluation...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy:.2f}")
    report = classification_report(y_test, y_pred)
    logging.info("Classification report:\n%s", report)
    print("Accuracy:", accuracy)
    print(report)

if __name__ == "__main__":
    logging.info("Script started.")
    
    try:
        # Load and preprocess data
        logging.info("Loading data...")
        df = load_data('../data/raw/e.csv')  # Update with your actual CSV file path
        X_train, X_test, y_train, y_test = preprocess_data(df)

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        evaluate_model(model, X_test, y_test)

        # Save the trained model for later use
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        logging.info("Model saved as 'model.pkl'.")

    except Exception as e:
        logging.error("An error occurred: %s", e)
    finally:
        logging.info("Script ended.")
