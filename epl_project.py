import streamlit as st
import tweepy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

#scrape tweets and return dataframe with new tweets
def scrape(tag, connection, number):
    #create dataframe for storing scraped tweets
    test_tweets = pd.DataFrame(columns =['text', 'tags_used'] )

    d = date.today()
    d = d.isoformat()
    tweets = tweepy.Cursor(connection.search, q = tag, lang = 'en', until = d,
    tweet_mode = 'extended').items(int(number))
    list_tweets = [tweet for tweet in tweets]
    i = 1
    #extract useful information from the tweets
    for tweet in list_tweets:
        hashtags = tweet.entities['hashtags']
        try:
            text = tweet.retweeted_status.full_text
        except AttributeError:
            text = tweet.full_text

        print('tweet:',text)
        hashtext = list()
        for j in range(0, len(hashtags)):
                hashtext.append(hashtags[j]['text'])
        #store tweet into the test dataframe
        ith_tweet = [text, hashtext]
        test_tweets.loc[len(test_tweets)]  = ith_tweet
        i = i+1
    if len(test_tweets) >=1:
        st.write('Got your data')
    else:
        st.write('Sorry, could not find anything with this tag')
    test_tweets.drop_duplicates(subset = ['text'], keep='last', inplace = True)
    test_tweets.reset_index(inplace=True, drop= True)
    return test_tweets

#show dataframe with scraped tweets
def show_test_dataframe(test_tweets):
    try:
        st.dataframe(test_tweets)
    except:
        st.write('Sorry, could not find the dataframe :(')
        st.write('Try scraping some data to fill up your test dataframe')

#tokenize the scraped tweets and return corpus
def tokenizer(frame):
    words_remove = ['ManchesterUnited', 'ManUnited', 'MUFC','mufc', 'manchester', 'liverpoolfc', 'liverpool', 'LFC', 'YNWA',
               'ynwa', 'lfc', 'Liverpool', 'Arsenal', 'AFC','COYG', 'arsenal','afc','Tottenham', 'tottenham', 'THFC', 'thfc','spurs',
               'SPURS','Everton','everton', 'EFC', 'ManchesterCity', 'manchestercity', 'city', 'Mancity', 'mancity', 'MCFC', 'mcfc',
               'LeicesterCity', 'leicestercity', 'leicester', 'LCFC', 'lcfc', 'CrystalPalace', 'crystalpalace', 'palace', 'CPFC', 'cpfc'
               , 'Chelsea', 'chelsea', 'ChelseaFC', 'CFC', 'cfc', 'SouthamptonFC', 'southampton', 'Southampton', 'Burnley', 'BurnleyFC',
               'burnley','Bournemouth', 'bournemouth', 'United', 'united']
    corpus = []
    for i in range(0,len(frame)):

        try:
            txt = re.sub(r"http\S+" , " ", frame['text'][i])
        except:pass
        try:
            txt = re.sub(r'@[A-Za-z0-9]+', ' ', txt)
        except:pass
        txt = txt.replace('#','')
        txt = re.sub('[^a-zA-Z]', ' ', txt)
        txt = txt.lower()
        txt = txt.split()
        temp = txt
        for w in list(temp):
            if w in words_remove:
                txt.remove(w)
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        txt = [ps.stem(word) for word in txt if word not in set(all_stopwords)]
        txt = ' '.join(txt)
        corpus.append(txt)

    return corpus

#connect to twitter api and return api object as it is used in scrape function
def connect_to_twitter():
        consumer_key = st.secrets['consumer_key']
        consumer_secret = st.secrets['consumer_secret']
        access_token = st.secrets['access_token']
        access_secret = st.secrets['access_secret']
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)
        api = tweepy.API(auth)

        return api

#make sentiment predictions for the preprocessed tweets
def predict(corpus):
    loaded_model = pickle.load(open('./model.sav', 'rb'))
    vectorizer = './vecorizer.pkl'
    loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(vectorizer, "rb")))
    new_corpus = loaded_vec.transform(corpus)
    pred = loaded_model.predict(new_corpus)

    return pred

@st.cache
def get_predicted_dataframe(tweet_data, predictions):

    p= pd.DataFrame(data= predictions, columns = ['prediction'])
    new_df = pd.concat([tweet_data, p], axis = 1, ignore_index = True)
    new_df.rename(columns = {0:'text',1:'tags used',2:'prediction'}, inplace=True)
    new_df['prediction'].replace(to_replace = '2', value = 'neutral', inplace=True)
    new_df['prediction'].replace(to_replace = '3', value = 'positive', inplace=True)
    new_df['prediction'].replace(to_replace = '1', value = 'negative', inplace=True)
    return new_df

def visualize(predictions):

    plot_data = pd.DataFrame(data= predictions, columns = ['prediction'])
    plot_data['prediction'].replace(to_replace = '2', value = 'neutral', inplace=True)
    plot_data['prediction'].replace(to_replace = '3', value = 'positive', inplace=True)
    plot_data['prediction'].replace(to_replace = '1', value = 'negative', inplace=True)
    values = plot_data['prediction'].value_counts().to_frame()
    st.markdown('### * interactive plot *')
    st.bar_chart(data= values)

@st.cache
def custom_prediction(text):
    words_remove = ['ManchesterUnited', 'ManUnited', 'MUFC','mufc', 'manchester', 'liverpoolfc', 'liverpool', 'LFC', 'YNWA',
               'ynwa', 'lfc', 'Liverpool', 'Arsenal', 'AFC','COYG', 'arsenal','afc','Tottenham', 'tottenham', 'THFC', 'thfc','spurs',
               'SPURS','Everton','everton', 'EFC', 'ManchesterCity', 'manchestercity', 'city', 'Mancity', 'mancity', 'MCFC', 'mcfc',
               'LeicesterCity', 'leicestercity', 'leicester', 'LCFC', 'lcfc', 'CrystalPalace', 'crystalpalace', 'palace', 'CPFC', 'cpfc'
               , 'Chelsea', 'chelsea', 'ChelseaFC', 'CFC', 'cfc', 'SouthamptonFC', 'southampton', 'Southampton', 'Burnley', 'BurnleyFC',
               'burnley','Bournemouth', 'bournemouth', 'United', 'united']
    #preprocessing of text
    corpus_new =[]
    text = text.replace('#','')
    text = re.sub(r"http\S+" , " ", text)
    text = re.sub(r'@[A-Za-z0-9]+', ' ', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    temp = text
    for w in list(temp):
        if w in words_remove:
            text.remove(w)
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    text = [ps.stem(word) for word in text if word not in set(all_stopwords)]
    text = ' '.join(text)
    corpus_new.append(text)

    res = predict(corpus_new)

    return res


def main():

    #APP LAYOUT
    st.title('Football tweets sentiment analysis')
    st.markdown('#### Made by Parth Munjal')

    option_one = 'Get data from twitter'
    option_two = 'Make a custom prediction'
    option_three = 'Home'
    prediction_method = st.sidebar.selectbox('Pick your method:',(option_three, option_two, option_one))
    if prediction_method == option_three:
        st.write('''> Choose one option from the side bar to predict
        sentiment in either tweets fetched from Twitter or a custom
        tweet written by you''')

        with st.beta_container():
            st.image('https://img.maximummedia.ie/joe_co_uk/eyJkYXRhIjoie1widXJsXCI6XCJodHRwOlxcXC9cXFwvbWVkaWEtam9lY291ay5tYXhpbXVtbWVkaWEuaWUuczMuYW1hem9uYXdzLmNvbVxcXC93cC1jb250ZW50XFxcL3VwbG9hZHNcXFwvMjAxNlxcXC8wN1xcXC8yNzE3MTgyNlxcXC90d2l0ZmlyLmpwZ1wiLFwid2lkdGhcIjo3NDAsXCJoZWlnaHRcIjo0MTYsXCJkZWZhdWx0XCI6XCJodHRwczpcXFwvXFxcL3d3dy5qb2UuY28udWtcXFwvYXNzZXRzXFxcL2ltYWdlc1xcXC9qb2Vjb3VrXFxcL25vLWltYWdlLnBuZz9pZD00OTdhNzlhOGM1MjRmYjVlYzYxOVwiLFwib3B0aW9uc1wiOltdfSIsImhhc2giOiI5NmRiMTE3NzA3ZWFlZDlkNzIyMTEzNTQzYWYyYmQ5YTliOTA0OTI4In0=/twitfir.jpg')

        st.markdown('### Go on, start predicting!')


    elif prediction_method == option_two:

        st.header('>> Custom tweet prediciton')
        st.write('''Here you can create your own tweet and test
        how the prediction model works on these custom made tweets''')

        #take input from user
        input  = st.text_area('Write your own tweet here (Try to make it football related)',
        value= '''That was a brilliant performance from Manchester United in the semi-final
        #mufc #manchesterunited #ggmu @brunofernandes @paulpogba''', max_chars = 280)
        st.write('You can also use the default value if you do not feel like writing')
        if st.button('Predict'):
            if not input:
                st.error('Please enter some text!')
                st.stop()
            else:
                result = custom_prediction(input)
                if result == '1':
                    st.markdown('### Prediction: Negative')
                elif result == '2':
                    st.markdown('### Prediction: Neutral')
                else:
                    st.markdown('### Prediction: Positive')
    else:
        #GET INPUT FROM USER
        st.header('>> Twitter data predictions')
        st.markdown('## Step 1: Enter a football related keyword/tag to search for')
        team = st.text_input(label = '''Enter value below:
        (or use our default value)''', value = 'Manchester United')
        #convert to lower case and remove spaces
        #for consistency in generating hashtags
        team = team.lower()
        team = re.sub(' ','', team)

        number_of_tweets = st.select_slider('Number of tweets to extract: (default is 10)'
        , options=[10, 20, 50, 100], value=10)
        st.info('available values are 10, 20, 50, 100')


        #
        #START PROCESS OF PREDICTION AFTER PREDICT IS CLICKED
        if st.button('Predict'):
            if not team:
                st.error('Please enter a team first!')
                st.stop()
            else:
                st.markdown('## Step 2: Scrape new tweets with Twitter API')
                #connect to api and fetch data
                st.write('Fetching some fresh tweets..')
                API = connect_to_twitter()
                tweets_frame = scrape(team, API, number_of_tweets)
                st.success('Ready!')

                st.markdown('### New tweets:')
                show_test_dataframe(tweets_frame)
                st.info('This dataframe might have lesser values than expect due to removal of duplicate tweets')

                #preprocess the tweets and ask if user wants the corpus to display
                st.markdown('## Step 3: Preprocessing the tweets for predictions')
                tweet_corpus = tokenizer(tweets_frame)
                st.success('Done!')

                #predict outcomes and show dataframe of tweets and their predictions
                st.markdown('## Step 4: Predicting outcomes for each tweet')
                st.markdown('#### Sentiments: Positive/neutral/Negative')
                predictions = predict(tweet_corpus)
                dataframe_with_predictions = get_predicted_dataframe(tweets_frame,predictions )

                st.dataframe(dataframe_with_predictions)

                #visualization of final data
                #code for filtering out tweets by sentiments
                st.markdown('## Step 5: Plotting')
                visualize(predictions)

if __name__ == "__main__":
    main()
