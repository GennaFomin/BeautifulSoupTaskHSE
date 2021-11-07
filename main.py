import pandas as pd
from bs4 import BeautifulSoup
import math
import argparse
import requests
from natasha import (
    Segmenter,
    MorphVocab,
    NewsNERTagger,
    NewsEmbedding,
    NewsMorphTagger,

    Doc
)

headers = {
    "User-Agent": "Mozilla/5.0 \
           (Macintosh; Intel Mac OS X 10_10_1) \
           AppleWebKit/537.36 (KHTML, like Gecko) \
           Chrome/39.0.2171.95 Safari/537.36"
}

divisions = ['backend', 'frontend', 'apps', 'software', 'testing', 'administration', 'design',
             'management', 'marketing', 'analytics', 'sale', 'content', 'support', 'hr', 'telecom',
             'other', 'office', 'security']

Description = pd.read_pickle('jobDescription')
Links = pd.read_pickle('linksDict')
Vectors = pd.read_pickle('vectorizedSet') # change vectorizedSet to vectorizedNamesOnly if you wish to work with names only
Words = pd.read_pickle('wordDict')

parser = argparse.ArgumentParser(description='Enter link on vacancy page')
parser.add_argument('link', type=str, help='Enter link on vacancy card')
link = parser.parse_args().link

# print(link)

request = requests.get(link, headers=headers)
soup = BeautifulSoup(request.content, 'html.parser')

if not request.ok:
    print('Something went wrong with your link...')
else:
    divisionRaw = soup.find('div', 'basic-section').a['href']
    division = ''
    for div in divisions:
        if divisionRaw.find(div) != -1:
            division = divisionRaw[divisionRaw.find(div):len(divisionRaw)]

    description = soup.find('div', "job_show_description__vacancy_description").find('div', 'style-ugc').text
    doc = Doc(description)
    emb = NewsEmbedding()

    morph_vocab = MorphVocab()
    morph_tagger = NewsMorphTagger(emb)
    segmentr = Segmenter()

    doc.segment(segmentr)
    doc.tag_morph(morph_tagger)

    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    doc.tag_morph(morph_tagger)
    preProcessedBoW = []

    for token in doc.tokens:
        if token.pos != 'PUNCT' and token.pos != 'PRON' and token.pos != 'CCONJ' and token.pos != 'DET' and token.pos != 'ADP':
            preProcessedBoW.append(token.lemma)

    descriptionVector = []
    listOfWords = Words[0].to_list()
    for token in listOfWords:
        if token in preProcessedBoW: # change to LemmaWords if all words
            descriptionVector.append(1)
        else:
            descriptionVector.append(0)

def cosineSimilarity(vectorA, vectorB):
    if len(vectorA) == len(vectorB):
        summ = 0
        squaredSumA = 0
        squaredSumB = 0
        for i in range(0, len(vectorB)):
            summ += vectorA[i]*vectorB[i]
            squaredSumA += vectorA[i]**2
            squaredSumB += vectorB[i]**2
    else:
        return 0
    if not squaredSumA*squaredSumB == 0:
        value = summ / (math.sqrt(squaredSumA)*math.sqrt(squaredSumB))
        return value
    else:
        return 0


if not descriptionVector == []: # if you want to work with all dict - lemmaWords, else - descriptionVector
    vectorList = Vectors[division].to_list()
    difference = {}
    for i in range(0, len(vectorList)):
        vector = vectorList[i]
        if len(vector) != 0 and len(descriptionVector) != 0:
            difference[i] = cosineSimilarity(vector, descriptionVector)
        else:
            difference[i] = 0
else:
    print('Some mistake occurred during code execution')

sortedAns = sorted(difference.items(), key=lambda x: -x[1])
print(sortedAns[:5])

baseLink = 'https://career.habr.com{}'
linksList = Links[division].to_list()
ansList = []
for i in range(0, 5):
    link = baseLink.format(linksList[sortedAns[i][0]])
    request = requests.get(link, headers=headers)
    if not request.ok:
        desList = Description[division].to_list()
        print(desList[i])
    else:
        print(link)
