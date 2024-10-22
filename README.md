# Twitter Sentiment Analysis using NLP

## Introduction 

Twitter sentiment analysis analyzes the sentiment or emotion of tweets. It uses natural 
language processing and machine learning algorithms to classify tweets automatically as 
positive, negative, or neutral based on their content. It can be done for individual tweets or 
a larger dataset related to a particular topic or event.

Importance of TWITTER SENTIMENTAL ANALYSIS:
- Understanding Customer Feedback: By analyzing the sentiment of customer feedback, companies can identify areas where they need to improve their products or services.
- Reputation Management: Sentiment analysis can help companies monitor their brand reputation online and quickly respond to negative comments or reviews.
- Political Analysis: Sentiment analysis can help political campaigns understand public opinion and tailor their messaging accordingly.
- Crisis Management: In the event of a crisis, sentiment analysis can help organizations monitor social media and news outlets for negative sentiment and respond appropriately.
- Marketing Research: Sentiment analysis can help marketers understand consumer behavior and preferences, and develop targeted advertising campaigns


## Objective 
To perform Twitter sentiment analysis using machine learning algorithms, I will utilize the Sentiment140 dataset to analyze the sentiment of tweets. A machine learning pipeline will be developed, incorporating Logistic Regression as the classification algorithm. The goal is to classify tweets into positive or negative sentiments.


## Data Overview

The dataset provided is the Sentiment140 Dataset which consists
of 1,600,000 tweets that have been extracted using the Twitter API. 

Link : https://www.kaggle.com/datasets/kazanova/sentiment140/code?datasetId=2477

The various columns present in this Twitter data are:
- target: the polarity of the tweet (positive or negative)
- ids: Unique id of the tweet
- date: the date of the tweet
- flag: It refers to the query. If no such query exists, then it is NO QUERY.
- user: It refers to the name of the user that tweeted
- text: It refers to the text of the tweet

## Python Packages Used 
-  Pandas : A Python library for data manipulation and analysis, providing powerful data structures like Data Frame for handling tabular data efficiently.
-  Matplotlib : A plotting library for creating static, interactive, and animated visualizations in Python, widely used for data visualization tasks.
- NumPy : A fundamental package for scientific computing in Python, providing support for arrays, matrices, and mathematical functions to efficiently manipulate large datasets.
-  re : A module in Python providing support for regular expressions, allowing for efficient string manipulation and pattern matching.
-  nltk : The Natural Language Toolkit is a Python library for working with human language data, providing tools and resources for tasks like tokenization, stemming, tagging, and parsing.
-  stopwords : A module in NLTK providing a list of common stopwords for various languages, used in text processing tasks like sentiment analysis and text classification to filter out irrelevant words.

## Steps Involved in Sentiment Analysis
###  Data Collection
- Obtain the Dataset: In this case, we are using the Sentiment140 dataset, which contains tweets labeled with either positive or negative sentiment.
- Explore the Dataset: Review the dataset structure (e.g., text and labels), number of samples, and any specific characteristics of the data.
### Data Preprocessing
- Text Cleaning : Remove unnecessary elements like URLs, mentions (@), hashtags, emojis, punctuation, and stopwords. Convert all text to lowercase to ensure uniformity.
- Tokenization : Break down the tweets into individual words or tokens.
- Lemmatization/Stemming : Reduce words to their base or root forms to simplify the vocabulary (e.g., "running" to "run").
- Handling Noise : Remove noisy or irrelevant data, such as excessively short tweets or tweets in languages other than English.
### Feature Extraction
- Text Vectorization: Convert the text data into numerical representations that can be understood by the machine learning model.
- Common techniques include:
   - Bag of Words (BoW): Represents the occurrence of words in the text.
   - TF-IDF (Term Frequency-Inverse Document Frequency): Measures the importance of a word in a document relative to its frequency across the dataset.
   - Word Embeddings (Optional): Use pre-trained embeddings like Word2Vec or GloVe for better word representations.
### Train-Test Split
Split the Dataset: Divide the dataset into training and testing sets (e.g., 80% for training and 20% for testing). This ensures that the model is evaluated on unseen data after training.
### Model Selection
Choose the Algorithm: For this project, Logistic Regression is chosen as the classification algorithm due to its simplicity and effectiveness for binary classification problems.
### Model Training
Train the Model: Fit the Logistic Regression model to the training data. The model will learn the relationship between the features (tweets) and the labels (positive or negative sentiment).
### Model Evaluation
Evaluate the Model: Test the trained model on the test dataset and evaluate its performance using metrics such as:
 - Accuracy: The percentage of correctly classified tweets.
 - Precision, Recall, and F1-Score: Additional metrics to assess the quality of the classification.
 - Confusion Matrix: A table that visualizes the true positive, true negative, false positive, and false negative classifications.


     - Accuracy Score on the training data is 0.8101671875
     - Accuracy Score on the test data is 0.77800625
     - Confusion Matrix :
    
       [ [121437  38563]

       [32475   127525] ]

## Interpretation 
Accuracy Score:
 - The accuracy score on the training data is approximately 0.8102, indicating that the model correctly predicts the sentiment of about 81.02% of the training instances.
- The accuracy score on the test data is approximately 0.7780, indicating that the model correctly predicts the sentiment of about 77.80% of the test instances.
- The slightly lower accuracy on the test data compared to the training data suggests some degree of overfitting, but the model still performs reasonably well on unseen data.

Confusion Matrix:

- True Negative (TN): 121437 instances were correctly classified as negative sentiment.
- False Positive (FP): 38563 instances were incorrectly classified as positive sentiment.
- False Negative (FN): 32475 instances were incorrectly classified as negative sentiment.
- True Positive (TP): 127525 instances were correctly classified as positive sentiment.





