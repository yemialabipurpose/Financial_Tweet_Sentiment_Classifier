# Financial Tweet Sentiment Classifier

This project aims to build a sentiment classifier for financial tweets using various machine learning and natural language processing techniques. The goal is to create a model that can accurately identify the sentiment of financial tweets as either positive, negative, or neutral based on their textual content.

## Dataset

The dataset used in this project was obtained from Hugging Face's datasets library. The dataset was already split into two CSV files: one for training with approximately 9000 rows and another for validation with about 2000 rows. Each file contains two columns: 'text' and 'label', where 'text' contains the raw tweet text, and 'label' represents the sentiment label (0 for negative, 1 for positive, and 2 for neutral).

## Data Preprocessing

Before feeding the data into machine learning models, data preprocessing was performed to clean and transform the text data into a suitable format. The following steps were applied to the 'text' column of the training data:

1. Removal of URLs: URLs present in the text were removed using regular expressions.

2. Removal of special characters: Non-word characters were removed to retain only meaningful words.

3. Lowercasing: All text was converted to lowercase to ensure uniformity.

4. Removal of stopwords: Commonly occurring English stopwords were removed to reduce noise in the data.

5. Stemming: The Porter stemming algorithm was applied to convert words to their root form.

The preprocessed text was then stored in two new columns: 'cleaned_text' and 'processed_text'.

## Model Training

To create the sentiment classifier, three different encoding approaches were explored, and three separate models were trained:

1. **Using BERT Model (BERT.ipynb):** The BERT (Bidirectional Encoder Representations from Transformers) tokenizer was used to tokenize the text and convert it into numerical representations. The pre-trained BERT model was utilized to obtain embeddings, and a linear classifier was trained on top of BERT embeddings to classify sentiments.

2. **Using TF-IDF Vectorizer (TF-IDF_Vectorizer.ipynb):** The processed text was encoded using the Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer. A linear support vector machine (SVM) classifier was trained on the TF-IDF features to predict sentiment labels.

3. **Using Word2Vec (Word2Vec.ipynb):** The processed text was tokenized, and Word2Vec embeddings were obtained for each word in the tweet. The tweet embeddings were created by taking the mean of all word embeddings in the tweet. A linear SVM classifier was trained on these tweet embeddings to perform sentiment classification.

## Results

The results obtained from each encoding approach were as follows:

### BERT Model:

- Accuracy: 0.71
| Class (Sentiment) | Precision | Recall | F1-Score | Support |
|-------------------|---------- |--------|----------|---------|
| 0 (Negative)      |   0.46    |  0.41  |   0.43   |  285    |
| 1 (Positive)      |   0.55    |  0.43  |   0.49   |  391    |
| 2 (Neutral)       |   0.79    |  0.87  |   0.83   |  1233   |
|-------------------|-----------|--------|----------|---------|
| Accuracy          |           |        |   0.71   |  1909   |
|-------------------|-----------|--------|----------|---------|
| Macro Avg         |   0.60    |  0.57  |   0.58   |  1909   |
|-------------------|-----------|--------|----------|---------|
| Weighted Avg      |   0.69    |  0.71  |   0.70   |  1909   |
|-------------------|-----------|--------|----------|---------|

### TF-IDF Vectorizer:

- Accuracy: 0.74

| Class (Sentiment) | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| 0 (Negative)      | 0.89      | 0.18   | 0.29     | 285     |
| 1 (Positive)      | 0.75      | 0.38   | 0.50     | 391     |
| 2 (Neutral)       | 0.73      | 0.98   | 0.84     | 1233    |
|-------------------|-----------|--------|----------|---------|
| Accuracy          |           |        | 0.74     | 1909    |
|-------------------|-----------|--------|----------|---------|
| Macro Avg         | 0.79      | 0.51   | 0.55     | 1909    |
|-------------------|-----------|--------|----------|---------|
| Weighted Avg      | 0.76      | 0.74   | 0.69     | 1909    |
|-------------------|-----------|--------|----------|---------|


### Word2Vec:

- Accuracy: 0.66 
| Class (Sentiment) | Precision | Recall | F1-Score | Support |
|-------------------|---------- |--------|----------|---------|
| 0 (Negative)      |   0.00    |  0.00  |   0.00   |  285    |
| 1 (Positive)      |   0.50    |  0.03  |   0.06   |  391    |
| 2 (Neutral)       |   0.65    |  1.00  |   0.79   |  1233   |
|-------------------|-----------|--------|----------|---------|
| Accuracy          |           |        |   0.65   |  1909   |
|-------------------|-----------|--------|----------|---------|
| Macro Avg         |   0.38    |  0.34  |   0.28   |  1909   |
|-------------------|-----------|--------|----------|---------|
| Weighted Avg      |   0.52    |  0.65  |   0.52   |  1909   |
|-------------------|-----------|--------|----------|---------|

## Conclusion

## Conclusion

Based on the results obtained from each encoding approach, we can draw the following conclusions:

### BERT Model:

- The BERT Model achieved an accuracy of 0.71, which indicates that it correctly predicted the sentiment of financial tweets in approximately 71% of the cases.
- The F1-scores for the three sentiment classes are relatively balanced, with the neutral class (F1-score: 0.83) outperforming the negative (F1-score: 0.43) and positive (F1-score: 0.49) classes.
- The weighted average F1-score of 0.70 suggests that the BERT Model performs reasonably well across all classes, considering class imbalances.
- The model's performance can be further improved by fine-tuning hyperparameters and increasing the training data.

### TF-IDF Vectorizer:

- The TF-IDF Vectorizer achieved an accuracy of 0.74, indicating slightly better performance than the BERT Model, correctly classifying 74% of the financial tweets.
- The F1-scores for the neutral class (F1-score: 0.84) and positive class (F1-score: 0.50) are relatively higher than the negative class (F1-score: 0.29).
- The weighted average F1-score of 0.69 indicates that the TF-IDF Vectorizer performs well but might struggle with class imbalances.
- The model's performance can be further improved by experimenting with different tokenization techniques and exploring more advanced text preprocessing techniques.

### Word2Vec:

- The Word2Vec approach achieved an accuracy of 0.66, which is slightly lower than both the BERT Model and the TF-IDF Vectorizer.
- The F1-score for the neutral class is relatively high (F1-score: 0.79), while the positive class has the lowest F1-score (F1-score: 0.06).
- The weighted average F1-score of 0.52 indicates that the Word2Vec approach faces challenges with the class imbalances and predicting the positive sentiment.
- The model's performance can be further enhanced by exploring different word embeddings, experimenting with context window size, and considering a more robust classifier.

In conclusion, all three encoding approaches show some level of success in predicting the sentiment of financial tweets, with the TF-IDF Vectorizer achieving the highest accuracy. However, there is room for improvement in all approaches, especially in handling class imbalances and addressing the challenges of predicting the positive sentiment class. Fine-tuning the models and experimenting with different configurations could lead to more accurate sentiment classification in financial tweets. Further experimentation and fine-tuning of models could potentially lead to even better results.

Feel free to explore the individual Jupyter Notebooks for more detailed implementation and analysis of each encoding approach.

*If you have any questions or feedback, please don't hesitate to reach out!*
