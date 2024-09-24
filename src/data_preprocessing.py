# data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pickle

def load_data(file_path):
    """Load data from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Preprocess the data by splitting and vectorizing."""
    # Check for NaN values and handle them
    df = df.dropna(subset=['description'])  # Remove rows where 'description' is NaN
    # Alternatively, you could fill NaN values with an empty string:
    # df['description'].fillna('', inplace=True)

    X = df['description']
    y = df['categories']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorizing the descriptions
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Save the vectorizer for later use
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    return X_train_vectorized, X_test_vectorized, y_train, y_test

if __name__ == "__main__":
    # Example usage
    df = load_data('../data/raw/e.csv')  # Update with your actual CSV file path
    X_train, X_test, y_train, y_test = preprocess_data(df)
