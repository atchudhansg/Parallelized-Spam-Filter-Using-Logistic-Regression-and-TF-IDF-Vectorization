import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

def train_spam_filter(data_path):
    try:
        # Load the data from the CSV file
        raw_mail_data = pd.read_csv(data_path)

        # Separate the data into text and labels
        X = raw_mail_data['text']
        Y = raw_mail_data['spam']

        # Split the data into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

        # Create a TF-IDF vectorizer
        feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

        # Transform the text data into feature vectors
        X_train_features = feature_extraction.fit_transform(X_train)
        X_test_features = feature_extraction.transform(X_test)

        # Convert labels to integers
        Y_train = Y_train.astype(int)
        Y_test = Y_test.astype(int)

        # Create a logistic regression model
        logistic_model = LogisticRegression()

        # Create a pipeline with feature extraction and the model
        spam_filter_model = Pipeline([('tfidfvectorizer', feature_extraction), ('logisticregression', logistic_model)])

        # Train the model
        spam_filter_model.fit(X_train, Y_train)

        # Calculate accuracy on test data
        Y_test_pred = spam_filter_model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_test_pred)
        print(f'Accuracy on test data: {accuracy * 100:.2f}%')

        return spam_filter_model

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    # Provide the path to the new dataset CSV file
    data_path = "F:\\emails.csv"
    spam_filter_model = train_spam_filter(data_path)

    if spam_filter_model:
        s = input("Enter the email: ")
        input_mail = [s]

        prediction = spam_filter_model.predict(input_mail)

        if prediction[0] == 1:
            print('Spam mail')
        else:
            print('Not spam mail')
