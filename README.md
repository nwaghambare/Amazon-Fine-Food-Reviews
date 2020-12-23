# Amazon-Fine-Food-Reviews
Project for sentiment analysis of Amazon Fine Food Reviews using NLP techniques and Logistic Regression.

This dataset is available on Kaggle. I have used sqlite3 in python to load the data.
The preprocessing is done in python itself. The reviews had a lot of duplicates and also large amount of reviews not related to food.
Many reviews contained HTML tags. All these were removed using regular expressions.

I have used TFIDF vectorization and stemming as part of encoding the text feature.
I tried a Naive Bayes model and a Logistic regression model. I got better results with the Logistic Regression model.

I have then used Flask to create a web server on which this model is deployed which takes in a review and provides a sentiment score (probability score) for the review being positive or negative.
