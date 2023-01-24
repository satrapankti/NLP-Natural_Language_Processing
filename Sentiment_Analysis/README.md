# Amazon Review Sentiment Analysis

### Amazon_Review_Sentiment_Analysis_GradientBoosting.py 
file has model based on Gradient Boosting Classifier Algorithm where the sentiments have a class imbalance where it has been solved using label encoder where sentiment of nearby polarity scores have been concatenated.

### Amazon_Review_Sentiment_Analysis_XGBoost.py 
file has model based on XGBoost Classifier where the problem of class imbalance is solved without concatination using sklearn.utils.class_weight where weights have been defined based on class sentiment.
