# Amazon Review Sentiment Analysis
![Amazon logo](https://user-images.githubusercontent.com/88264074/230715202-bca3968c-0d3b-45cb-95df-4ca86fd3fc81.png)

## Amazon Sentimental Analysis.ipynb
This file has EDA, Visualization, Model Building,Selection & Accuracy

## Amazon_Review_Sentiment_Analysis_GradientBoosting.py 
This file has model based on Gradient Boosting Classifier Algorithm where the sentiments have a class imbalance where it has been solved using label encoder where sentiment of nearby polarity scores have been concatenated. 

## Amazon_Review_Sentiment_Analysis_XGBoost.py 
This file has model based on XGBoost Classifier where the problem of class imbalance is solved without concatination using sklearn.utils.class_weight where weights have been defined based on class sentiment.

`Both of the above files are coded in a way which can predict sentiments as well as can be deployed using Streamlit`
