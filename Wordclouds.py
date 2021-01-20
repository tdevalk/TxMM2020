import pickle
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

with open('listfile.pickle', 'rb') as filehandle:
    # read the data as binary data stream
    keywords_and_emotions = pickle.load(filehandle)

years_index = np.arange(30, 36, 1).tolist()
fig, axs = plt.subplots(3, 2, figsize=(8, 8))
#fig.title(f'Wordcloud for year {year + 1960} and emotion {emotion}')

for emotion in ['anger']:
    for year in years_index:
        fullTermsDict = {}
        for element in keywords_and_emotions[year]:
            fullTermsDict[element['text']] = element['emotion'][emotion]
        wc = WordCloud(background_color='white', max_words=25)
        wc.generate_from_frequencies(fullTermsDict)
        axs[(year-30)%2 - (year-30)%3, (year-30)%2].imshow(wc, interpolation="bilinear")
        axs[(year-30)%2 - (year-30)%3, (year-30)%2].set_title(f'Wordcloud for {emotion} in {year + 1960}', fontsize=14)
        axs[(year-30)%2 - (year-30)%3, (year-30)%2].axis('off')
    fig.tight_layout(pad=1.0)
    plt.show()


years_index_2 = np.arange(20, 26, 1).tolist()
fig_2, axs_2 = plt.subplots(3, 2, figsize=(8, 8))

for emotion in ['joy']:
    for year in years_index_2:
        fullTermsDict = {}
        for element in keywords_and_emotions[year]:
            fullTermsDict[element['text']] = element['emotion'][emotion]
        wc = WordCloud(background_color='white', max_words=25)
        wc.generate_from_frequencies(fullTermsDict)
        axs_2[(year-20)%2 - (year-20)%3, (year-20)%2].imshow(wc, interpolation="bilinear")
        axs_2[(year-20)%2 - (year-20)%3, (year-20)%2].set_title(f'Wordcloud for {emotion} in {year + 1960}', fontsize=14)
        axs_2[(year-20)%2 - (year-20)%3, (year-20)%2].axis('off')
    fig_2.tight_layout(pad=1.0)
    plt.show()

years_index_3 = np.arange(30, 40, 1).tolist()
fig_3, axs_3 = plt.subplots(5, 2, figsize=(8, 8))
for emotion in ['joy']:
    for year in years_index_3:
        fullTermsDict = {}
        for element in keywords_and_emotions[year]:
            fullTermsDict[element['text']] = element['emotion'][emotion]
        wc = WordCloud(background_color='white', max_words=25)
        wc.generate_from_frequencies(fullTermsDict)
        axs_3[int((year-30)/2), (year-30)%2].imshow(wc, interpolation="bilinear")
        axs_3[int((year-30)/2), (year-30)%2].set_title(f'Wordcloud for {emotion} in {year + 1960}', fontsize=10)
        axs_3[int((year-30)/2), (year-30)%2].axis('off')
    fig_3.tight_layout(pad=1.0)
    plt.show()