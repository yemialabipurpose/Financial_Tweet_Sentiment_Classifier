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


## Model Training

For model training, I leveraged the following Python libraries:

1. **Gensim**: I used the `gensim` library to access pre-trained word embeddings. Specifically, I loaded the 'word2vec-google-news-300' model from the Gensim downloader. Word embeddings are dense vector representations of words that capture semantic relationships between words. These embeddings provided valuable features for the sentiment classifier.

2. **Scikit-learn**: The `scikit-learn` library was used for data splitting. I defined the `even_train_test_split()` function to ensure an even distribution of data across the three sentiment classes: negative, positive, and neutral. This function used the scikit-learn library's `train_test_split` function to split the data for each sentiment class separately, while maintaining the balance of samples in the training and testing sets.


## Model Evaluation and Selection

In this section, I will walk you through the task of model evaluation and selection to identify the most suitable classifier for sentiment analysis on financial tweets. My approach involved the following steps:

### Step 1: Defining Model Libraries and Hyperparameters

I started by importing the necessary model libraries, including classifiers such as Logistic Regression, Support Vector Machine (SVM), Naive Bayes, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), Stochastic Gradient Descent (SGD), XGBoost, AdaBoost, CatBoost, and Gradient Boosting. For each model, I defined a parameter grid containing various hyperparameters to be tuned during the model evaluation process.

### Step 2: Evaluating Models and Obtaining Accuracy Scores

With my model libraries and hyperparameters ready, I moved on to the evaluation step. I defined a function named `evaluate_models(X_train, y_train, X_test, y_test, models, params)` to evaluate each model's performance with the dataset. This function performed `GridSearchCV` to optimize hyperparameters and trained each model with the best parameters.

### Step 3: Comparing Model Accuracies

After training each model, I obtained the accuracy scores for the different model on the test datasets. The table below showcases the accuracy results for the various models I tried:

| Model                | Accuracy |
|----------------------|---------:|
| Logistic Regression  | 0.5062   |
| SVM                  | 0.5073   |
| Naive Bayes          | 0.4342   |
| Decision Tree        | 0.4454   |
| Random Forest        | 0.5062   |
| KNN                  | 0.4882   |
| SGD                  | 0.4882   |
| XgBoost              | 0.4938   |
| AdaBoost             | 0.4522   |
| CatBoost             | 0.5231   |
| Gradient Boosting    | 0.5028   |

### The Best Model: CatBoostClassifier

After comparing the model accuracies, the `CatBoostClassifier` emerged as the top-performing classifier with an accuracy score of 0.5231 as a result, it was selected. The reports tables below show the performance of the  `CatBoostClassifier model` on the test dataset.  

### Confusion Matrix

The confusion Matrix give insights into how the model performed. It tell you what proportion of the classes has the model classified rightly;

| Actual\Predicted | Negative | Positive | Neutral |
|------------------|---------:|---------:|--------:|
| **Negative (0)** |   118    |   87     |   84    |
| **Positive (1)** |   59     |   164    |   77    |
| **Neutral (2)**  |   64     |   53     |   183   |

### Classification Report

We obtained the classification report for the CatBoost model and summarized it in a tabular format as follows:

| Class (Sentiment) | Precision | Recall | F1-Score | Support |
|-------------------|----------:|------:|---------:|--------:|
| **Negative (0)**  |    0.49   |  0.41 |    0.45  |   289   |
| **Positive (1)**  |    0.54   |  0.55 |    0.54  |   300   |
| **Neutral (2)**   |    0.53   |  0.61 |    0.57  |   300   |
| **Accuracy**      |           |       |   0.52   |   889   |
| **Macro Avg**     |    0.52   |  0.52 |    0.52  |   889   |
| **Weighted Avg**  |    0.52   |  0.52 |    0.52  |   889   |


**Confusion Matrix Interpretation:**
- True Negative (TN): 118 instances were correctly classified as Negative.
- True Positive (TP): 164 instances were correctly classified as Positive.
- True Neutral (TN): 183 instances were correctly classified as Neutral.
- False Positive (FP): 87 instances were incorrectly classified as Positive when they were Negative.
- False Negative (FN): 59 instances were incorrectly classified as Negative when they were Positive, and 64 instances were incorrectly classified as Negative when they were Neutral, and 53 instances were incorrectly classified as Positive when they were Neutral.

## Validation of the Model with Validation Dataset

To validate the model's robustness, I evaluated its performance on the validation dataset. The validation dataset also has a skewed class distribution, but our primary focus was to assess how well the model generalizes to new data. Here are the results:

**Confusion Matrix (Validation)**

| Actual\Predicted | Negative | Positive | Neutral |
|------------------|---------:|---------:|--------:|
| **Negative (0)** |   154    |   116    |    77   |
| **Positive (1)** |   118    |   227    |   130   |
| **Neutral (2)**  |   301    |   300    |   965   |

**Classification Report (Validation)**

| Class (Sentiment) | Precision | Recall | F1-Score | Support |
|-------------------|----------:|------:|---------:|--------:|
| **Negative (0)**  |    0.29   |  0.44 |    0.33  |   347   |
| **Positive (1)**  |    0.35   |  0.48 |    0.41  |   575   |
| **Neutral (2)**   |    0.82   |  0.62 |    0.70  |   1566  |
| **Accuracy**      |           |       |   0.56   |   2388  |
| **Macro Avg**     |    0.48   |  0.51 |    0.48  |   2388  |
| **Weighted Avg**  |    0.65   |  0.56 |    0.59  |   2388  |

## Conclusion

In conclusion, the CatBoostClassifier performed well, achieving an accuracy of 52% on the test dataset and 56% on the validation dataset. While there is always room for improvement, these results show that our sentiment classifier model is promising and capable of providing valuable insights into market sentiment from financial tweets.
