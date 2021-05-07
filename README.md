# sentiment_analysis_app
Streamlit app to detect sentiments in football related tweets

In this project I have tried to analyse tweets about english premier league football teams to detect sentiments and then use the trained model to make future predictions.

## Dataset and data preprocessing

The dataset comprises of tweets related to teams in the english premier league. You can check out the model creation jupyter notebook in this repository to get a look at the dataset used for training

## Model training and selection

The tweets have been classified into three classes i.e positive, negative and neutral. I have used a logistic regression model with a one vs rest approach to detect multiple classes.  
I tried out several classification models and Logistic regression gave me the best results in terms of predicting multiple classes and of course accuracy.
A more detailed approach can again be seen in the model creation notebook in this repository

## About the streamlit application

The streamlit app uses the saved model from the epl tweets dataset to make predictions. The app is connected to the twitter API. You can enter a key word to search the API for extracting tweets with the hashtag you mentioned in the input box.
After the tweets are extracted appropriate preprocessing is applied and the model then provides its sentiment predictions for each tweet along with a an appropriate bar plot displaying the number of predictions for each class.

You can also use the app to make a single prediction i.e enter your own text/tweet in text box and make a sentiment prediction using my model. 

You can find the python script used to create the streamlit web application in this repo with the name epl_project.py
Also check out the model_creation jupyter notebook to understand my approach for preprocessing the training data, texts, training and finding the most effective model and also saving the model so it can be used in the web app.
