# Financial Tweet Sentiment Classifier

In this project, my main objective was to build a sentiment classifier for financial tweets using various machine learning and natural language processing techniques. I aimed to create a model that could accurately identify the sentiment of financial tweets as either positive, negative, or neutral based on their textual content.

## Dataset

For this project, I obtained the dataset from Hugging Face's datasets library. The dataset was already split into two CSV files: one for training with approximately 9000 rows and another for validation with about 2000 rows. Each file contains two columns: 'text' and 'label'. The 'text' column contains the raw tweet text, and the 'label' column represents the sentiment label (0 for negative, 1 for positive, and 2 for neutral).

## Exploratory Data Analysis

Before diving into preprocessing and training, I conducted some exploratory data analysis (EDA) to understand the dataset's distribution and balance. To visualize the sentiment class distribution, I used the seaborn library to create a count plot of the sentiment labels. The plot showed that the data was imbalanced, with label 2 (Neutral) having approximately three times as many data samples as label 0 (Negative) and label 1 (Positive).

![Sentiment Class Distribution](https://github.com/yemialabipurpose/Financial_Tweet_Sentiment_Classifier/assets/37623664/6485f2a0-e6ea-4c5f-b49b-0dfed2fa8aca)

## Data Balancing

To address the data imbalance, I balanced the number of samples for each sentiment class. The goal was to have approximately equal numbers of samples for each class. After balancing the dataset, each sentiment class contained roughly the same number of samples, with 1500 samples for Neutral and Positive classes and 1442 samples for the Negative class. This created a more suitable foundation for training a sentiment analysis model.

The final balanced dataset had the following distribution:

- Neutral: 1500 samples
- Positive: 1500 samples
- Negative: 1442 samples
- Total: 4442 samples

## Data Preprocessing

To prepare the text data for NLP (Natural Language Processing), I utilized several Python libraries to perform the following preprocessing steps:

1. **URL Removal**: I used regular expressions and the `re` library to remove any URLs present in the tweet text. The function was named `preprocess()`.

2. **Special Characters and Numbers Removal**: The same `preprocess()` function was used to remove special characters and numbers from the text.

3. **Stock Symbols Removal**: The `preprocess()` function was modified to remove stock symbols (e.g., $AAPL) from the text.

4. **Tokenization and Lemmatization**: For tokenization and lemmatization, I utilized the `spacy` library. The `preprocess()` function converted each tweet into a sequence of tokens and performed lemmatization to reduce each word to its base or root form.

5. **Stop Words Removal**: The `preprocess()` function was further enhanced to remove common stop words (e.g., "and", "the", "is") using the `spacy` library.


# Model Training

For model training, I leveraged the following Python libraries:

1. **Gensim**: I used the `gensim` library to access pre-trained word embeddings. Specifically, I loaded the 'word2vec-google-news-300' model from the Gensim downloader. Word embeddings are dense vector representations of words that capture semantic relationships between words. These embeddings provided valuable features for the sentiment classifier.

2. **Scikit-learn**: The `scikit-learn` library was used for data splitting. I defined the `even_train_test_split()` function to ensure an even distribution of data across the three sentiment classes: negative, positive, and neutral. This function used the scikit-learn library's `train_test_split` function to split the data for each sentiment class separately, while maintaining the balance of samples in the training and testing sets.


