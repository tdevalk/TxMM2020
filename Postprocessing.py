#import packages
from gensim.summarization import keywords
import spacy
import nltk
import lyricsgenius
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from textblob import TextBlob
from nltk.corpus import stopwords
from spacy.lang.en import English

import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions, KeywordsOptions

natural_language_understanding = NaturalLanguageUnderstandingV1(version='2020-08-01')

natural_language_understanding.set_service_url('https://api.eu-de.natural-language-understanding.watson.cloud.ibm.com/instances/cf0b1bf5-152f-4488-9fc2-62eaf689d38e')

#nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
nlp = English()
nlp.max_length = 20000000

def read_in_dataframes():
    for index in range(20):
        if index == 0:
            df = pd.read_pickle(f'Z:\TXMM_data\hot100data_{index}.pkl')
        else:
            dataframe = pd.read_pickle(f'Z:\TXMM_data\hot100data_{index}.pkl')
            df = df.append(dataframe, ignore_index=True)
    df = df[df['emotions'].notnull()]
    df = df.reset_index(drop=True)
    sadness = np.empty(len(df.index))
    joy = np.empty(len(df.index))
    fear = np.empty(len(df.index))
    disgust = np.empty(len(df.index))
    anger = np.empty(len(df.index))
    for index, row in df.iterrows():
        print(index)
        print(df['emotions'].iloc[index])
        sadness[index] = df['emotions'].iloc[index]['sadness']
        joy[index] = df['emotions'].iloc[index]['joy']
        fear[index] = df['emotions'].iloc[index]['fear']
        disgust[index] = df['emotions'].iloc[index]['disgust']
        anger[index] = df['emotions'].iloc[index]['anger']
    df['sadness'] = sadness
    df['joy'] = joy
    df['fear'] = fear
    df['disgust'] = disgust
    df['anger'] = anger
    return df

def get_lyric_sentiment(lyrics):
    '''
    Function to return sentiment score of each song
    '''
    analysis = TextBlob(lyrics)
    return analysis.sentiment.polarity

def get_lyric_emotions(lyrics):
    """
    Function to return emotion scores for each song
    """
    response = natural_language_understanding.analyze(
        text=lyrics, features=Features(emotion=EmotionOptions())).get_result()
    return response['emotion']['document']['emotion']

# Function to preprocess text
def preprocess(text):
    # Create Doc object
    doc = nlp(text, disable=['ner', 'parser'])
    # Generate lemmas
    lemmas = [token.lemma_ for token in doc]
    # Remove stopwords and non-alphabetic characters
    a_lemmas = [lemma for lemma in lemmas
                if lemma.isalpha() and lemma not in stopwords]
    return ' '.join(a_lemmas)


"""Extract Keywords from text"""
def return_keywords(texts):
    xkeywords = []
    values = keywords(text=preprocess(texts), split='\n', scores=True)
    for x in values[:10]:
        xkeywords.append(x[0])
    try:
        return xkeywords
    except:
        return "no content"

"""Extract Keywords for particular emotions from text"""
def return_keywords_and_emotion(texts):
    response = natural_language_understanding.analyze(
        text=texts,
        features=Features(keywords=KeywordsOptions(sentiment=True, emotion=True, limit=25))).get_result()
    return response['keywords']

def add_keyword_emotion_values(emotion_list):
    df= pd.DataFrame()
    df['sadness'] = [np.average([d['emotion']['sadness'] for d in year]) for year in emotion_list]
    df['joy'] = [np.average([d['emotion']['joy'] for d in year]) for year in emotion_list]
    df['fear'] = [np.average([d['emotion']['fear'] for d in year]) for year in emotion_list]
    df['disgust'] = [np.average([d['emotion']['disgust'] for d in year]) for year in emotion_list]
    df['anger'] = [np.average([d['emotion']['anger'] for d in year]) for year in emotion_list]
    return df


def return_lyrics_emotion_values(dataframe):
    sadness_values = dataframe['sadness'].resample('Y').mean()
    joy_values = dataframe['joy'].resample('Y').mean()
    fear_values = dataframe['fear'].resample('Y').mean()
    disgust_values = dataframe['disgust'].resample('Y').mean()
    anger_values = dataframe['anger'].resample('Y').mean()
    return {'sadness': sadness_values, 'joy': joy_values, 'fear': fear_values, 'disgust': disgust_values,
            'anger': anger_values}

def plot_emotion_over_time(emotions_values, xaxis):
    for emotion in ['sadness', 'joy', 'fear', 'disgust', 'anger']:
        plt.plot(xaxis, emotions_values[emotion], label=emotion)
    plt.legend()
    plt.show()



#Postprocessing

dataframe = read_in_dataframes()

#Resample daraframe lyrics by year. Get all the lyrics for every song for each year
dataframe['WeekID'] = pd.to_datetime(dataframe['WeekID'], format='%Y-%m-%d')
dataframe = dataframe.set_index(['WeekID'])
lyrics_resample = dataframe['Lyrics'].resample('Y').sum()

# Resample emotions by year. Get all the emotions for every song for each year
emotion_values = return_lyrics_emotion_values(dataframe)
plot_emotion_over_time(emotion_values, xaxis = emotion_values['sadness'].index)

#Get keywords and emotions associated with these keywords
keywords_plus_emotions = np.array([return_keywords_and_emotion(x[1]) for x in lyrics_resample.iteritems() if x[1] != 0])
print(keywords_plus_emotions)

# plot emotion values associated with keywords
df = add_keyword_emotion_values(keywords_plus_emotions)
print(df)

with open("listfile.pickle", "wb") as f:
    pickle.dump(keywords_plus_emotions, f)

