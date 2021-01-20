#import packages
from gensim.summarization import keywords
import spacy
import nltk
import lyricsgenius
import datetime
import pandas as pd
import numpy as np
from textblob import TextBlob
from nltk.corpus import stopwords
from spacy.lang.en import English

import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson import LanguageTranslatorV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions, KeywordsOptions

# authenticator = IAMAuthenticator('apikey:lcZGJ7q1bxL7ZjbaGkBfRv0gqnek1ScT5qA8Sn1DD-Qf')
# natural_language_understanding = NaturalLanguageUnderstandingV1(
#     version='2020-08-01',
#     authenticator=authenticator
# )

natural_language_understanding = NaturalLanguageUnderstandingV1(version='2020-08-01')
natural_language_translator = LanguageTranslatorV3(version='2018-05-01')

natural_language_understanding.set_service_url('https://api.eu-de.natural-language-understanding.watson.cloud.ibm.com/instances/cf0b1bf5-152f-4488-9fc2-62eaf689d38e')
natural_language_translator.set_service_url('https://api.eu-de.language-translator.watson.cloud.ibm.com/instances/488120ac-4ea8-4b1e-a393-d1d4c7bfb4a2')
# {
#   "apikey": "cr2R1qmDjXlYeSmbzz-MfvOTRGuBY37DSXBw6rwmRWMn",
#   "iam_apikey_description": "Auto-generated for key 7994271e-faf1-4464-932b-9c5577e604bf",
#   "iam_apikey_name": "Auto-generated service credentials",
#   "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Manager",
#   "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/de4c50f59dad4754a2de91308fc5a7b1::serviceid:ServiceId-fbfe9789-afc2-4f2e-8a16-eb3109871bc7",
#   "url": "https://api.eu-de.language-translator.watson.cloud.ibm.com/instances/488120ac-4ea8-4b1e-a393-d1d4c7bfb4a2"
# }

#nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

nlp = English()
nlp.max_length = 100000




#define Genius API authentication
genius = lyricsgenius.Genius('7x_jEYh_M6i4UgeK19y-gv7DPhRBycsJlwyHHGVUiLf-QSS82wkt1KOFqFfVn0jw')

#import billboard hot100 dataset
hot100_df = pd.read_csv('https://query.data.world/s/qf6et5c7dh23kglnvjcoztlmom62it')
hot100_df.drop_duplicates(subset='SongID', inplace = True) #remove duplicate occurrences of songs
hot100_df.reset_index()
print(f'length of dataset is: {len(hot100_df)}') #28474
#Filtering for years 1960-2014
hot100_df['WeekID'] = pd.to_datetime(hot100_df['WeekID'], format='%m/%d/%Y')
minimum = datetime.datetime.strptime('01/01/1960', '%m/%d/%Y')
maximum = datetime.datetime.strptime('01/01/2015', '%m/%d/%Y')
hot100_df = hot100_df[hot100_df['WeekID'] >= minimum]
hot100_df = hot100_df[hot100_df['WeekID'] < maximum]
print(f'length of dataset is: {len(hot100_df)}') #25170
hot100_sample = hot100_df
#Splitting in 10 parts
dataframe_parts = np.array_split(hot100_sample, 20)

def get_lyrics(title, artist):
    '''
    Function to return lyrics of each song using Genius API
    '''
    try:
        return genius.search_song(title, artist).lyrics
    except:
        return 'not found'


def get_lyric_sentiment(lyrics):
    '''
    Function to return sentiment score of each song
    '''
    try:
        analysis = TextBlob(lyrics)
        return analysis.sentiment.polarity
    except:
        return None

def get_lyric_emotions(lyrics):
    """
    Function to return emotion scores for each song
    """
    try:
        response = natural_language_understanding.analyze(
            text=lyrics, features=Features(emotion=EmotionOptions())).get_result()
        return response['emotion']['document']['emotion']
    except:
        return None

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
    xkeywords = []
    response = natural_language_understanding.analyze(
        text=texts,
        features=Features(keywords=KeywordsOptions(sentiment=True, emotion=True, limit=10))).get_result()
    return response['keywords']['emotions']

emotion_test = get_lyric_emotions("Qui dit étude dit travail Qui dit taf te dit les thunes Qui dit argent dit dépenses")
print(emotion_test)

for index, hot100_sample in enumerate(dataframe_parts):
    print(f'length of sample is: {len(hot100_sample)}')  #
    #Use get_lyrics funcion to get lyrics for every song in dataset
    lyrics = hot100_sample.apply(lambda row: get_lyrics(row['Song'], row['Performer']), axis =1)
    hot100_sample['Lyrics'] = lyrics
    hot100_sample = hot100_sample.drop(hot100_sample[hot100_sample['Lyrics'] == 'not found'].index)  # drop rows where lyrics are not found on Genius

    # Filter out songs that are not English

    hot100_sample['language'] = hot100_sample.apply(lambda row: json.loads(
        json.dumps(natural_language_translator.identify(row['Lyrics']).get_result(), indent=2))['languages'][0][
        'language'] if len(row['Lyrics'].encode('utf-8')) < 50000 else 'undecided', axis=1)
    hot100_sample = hot100_sample[hot100_sample['language'] == 'en']
    print(f'length of sample is: {len(hot100_sample)}')  #


    #Use get_lyric_sentiment to get sentiment score for all the song lyrics
    sentiment = hot100_sample.apply(lambda row: get_lyric_sentiment(row['Lyrics']), axis =1)
    hot100_sample['Sentiment'] = sentiment

    #Use get_lyric_emotions to get emotion scores for all the song lyrics
    emotions = hot100_sample.apply(lambda row: get_lyric_emotions(row['Lyrics']), axis =1)
    hot100_sample['emotions'] = emotions
    #print(hot100_sample['emotions'])

    #Storing data locally
    hot100_sample.to_pickle(f'Z:\TXMM_data\hot100data_{index}.pkl')

#skfeuh123**H