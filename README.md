# Twitter Sentimental Analysis using Naive Bayes
## Overview

This project focuses on sentiment analysis utilizing the Naive Bayes algorithm. Sentiment analysis involves discerning the sentiment expressed in textual data, which is crucial for various applications such as customer feedback analysis and social media sentiment monitoring.

## Methodology

1. **Data Collection:**
   - Twitter Sentiment Analysis Dataset is obtained from Kaggle.
  
2. **Data Pre-Processing:**
   - Removal of Twitter handles, dots, numbers, etc.
   - Elimination of short words (less than 2 letters).
   - Tokenization: Splitting each string into a list of words.
   - Stemming: Converting words to their root/base form to reduce dimensionality.

3. **Word Cloud:**
   - Visualization of most frequent words in tweets based on sentiment labels.

4. **Data Analysis:**
   - Extraction and analysis of hashtags from tweets based on sentiment labels.

5. **Model Building:**
   - Feature extraction using Count Vectorizer.
   - Splitting data into train and test sets.
   - Implementation of Multinomial Naive Bayes Classifier for probabilistic sentiment classification.

6. **Model Evaluation and Results:**
   - Evaluation of model accuracy using `accuracy_score`.
   - Interpretation of results through confusion matrix to understand model performance.

## Files Included

-  Twitter Sentiment Analysis Dataset
-  Python script for data pre-processing
-  Python script for generating word clouds
-  Python script for extracting hashtags and analysis
-  Python script for model building and evaluation

## Dependencies

- Python 3.x
- pandas
- numpy
- nltk
- sklearn

## Usage

1. Clone the repository:
   git clone
       https://github.com/Blackmonarch4574/Twitter-Sentimental-Analysis.git



