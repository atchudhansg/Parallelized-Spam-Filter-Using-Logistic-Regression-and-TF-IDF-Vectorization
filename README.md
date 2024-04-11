# Parallelized Spam Filter Using Logistic Regression and TF-IDF Vectorization

## Overview
This project implements a spam filter for email messages using machine learning techniques. The system utilizes logistic regression as the classification algorithm and TF-IDF vectorization for feature extraction from the text of email messages. Additionally, the project employs parallelization techniques to optimize the feature extraction process, improving the efficiency of the spam filter.

## Features
- **Logistic Regression Model:** Utilizes logistic regression for binary classification of email messages into spam and non-spam categories.
- **TF-IDF Vectorization:** Converts email text into numerical feature vectors using the TF-IDF (Term Frequency-Inverse Document Frequency) technique, capturing the importance of words in the documents.
- **Parallelization:** Utilizes parallel processing techniques to optimize the feature extraction process, enhancing the efficiency of the spam filter.
- **Scalable:** Can handle large datasets efficiently, making it suitable for real-world email filtering applications.

## Dependencies
- Python 3.x
- NumPy
- pandas
- scikit-learn
- joblib (for parallelization)

## Usage
1. Clone the repository to your local machine:

    ```
    git clone https://github.com/atchudhansg/Parallelized-Spam-Filter-Using-Logistic-Regression-and-TF-IDF-Vectorization.git

    ```

2. Navigate to the project directory:

    ```
    cd Parallelized Spam Filter Using Logistic Regression and TF-IDF Vectorization
    ```



3. Download the dataset provided here in the repository to train the machine learning model.

4. Update the `data_path` variable in the `train_spam_filter` function of `Parallelized Spam Filter Using Logistic Regression and TF-IDF Vectorization.py` with the path to your dataset CSV file.

5. Run the `Parallelized Spam Filter Using Logistic Regression and TF-IDF Vectorization.py` script to train the spam filter model and evaluate its performance:

    ```
    python Parallelized Spam Filter Using Logistic Regression and TF-IDF Vectorization.py
    ```

6. Follow the on-screen prompts to input an email text and test the spam filter's prediction.

## License
This project is free to be used and downloaded, as long as it remains relevant for educational purposes.

## Acknowledgements
- This project is inspired by the need for efficient email spam filtering techniques.
- The implementation builds upon concepts and techniques from machine learning and natural language processing.
- Thanks to the open-source community for providing libraries and tools that make projects like this possible.
- The dataset used for training the spam filter is sourced from Kaggle. The dataset, named `emails.csv`, contains 2 columns, one named 'text' which contains the email text, and the output column 'spam', which contains 2 categorical values- 'spam' or 'no spam'. I thank Kaggle for providing this valuable dataset, which has been instrumental in the development of my spam filter model.
