"""
Lab1 data preprocessing

@author: Zhe Dong, Kun Liu
"""
from __future__ import print_function
class Document:
	def __init__(self):
		self.title = ""
		self.topics = []
		self.places = []
		self.id = 0
		self.freq = {}
		self.freq_vector = []
  		self.tf_idf_vector = []


import nltk
import sys
sys.setrecursionlimit(100000)
import operator
from os import listdir
from bs4 import BeautifulSoup

stemmer = nltk.stem.porter.PorterStemmer() 
lemmatizer = nltk.stem.WordNetLemmatizer()
stop = nltk.corpus.stopwords.words("english")
Vocabulary = {}
Document_List = []


from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents(categories='news')
pos_tagger = nltk.UnigramTagger(brown_tagged_sents)

from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):
    if treebank_tag is None:
	return wordnet.NOUN
    elif treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def get_word_frequency(soup):
    currentFreq=[]
    for doc in soup.find_all("reuters"):
                D = Document()
		id = doc.get("newid")
                print ("parsing document #" + id)
                if doc.find("title") is not None:
			D.title = unicode(doc.find("title").string)
			#print D.title
            	if doc.find("topics") is not None:
                    for topic in doc.topics.children:                    
                        D.topics.append(unicode(doc.find("topics").string))
                if doc.find("places") is not None:
                    for place in doc.places.children:                    
                        D.places.append(unicode(doc.find("places").string))
		D.id = int(id)
		D.freq = {}
                if doc.find("body") is not None:
                        words = nltk.word_tokenize(doc.find("body").string)
                        words_and_pos = pos_tagger.tag(words)
			for word_pos in words_and_pos:
                                word = word_pos[0].lower()
				pos = get_wordnet_pos(word_pos[1])
				if word.endswith("'s"):
          				word = word[:-2]
				if (len(word) <= 1):
					continue
				if not word[0].isalpha():
					continue
				if word in stop:
					continue
                                if "." in word and len(word)==2:
                                    	continue
                                word = lemmatizer.lemmatize(word, pos)
                                word = stemmer.stem(word) #get words stemmed
				if word not in D.freq:
                                        if word not in Vocabulary:
                                                Vocabulary[word] = 1
                                        else:
                                                Vocabulary[word] += 1
                                        D.freq[word] = 1
                                else:
                                        D.freq[word] += 1
    		currentFreq.append(D)
    return currentFreq
    
#ask for directory for dataset
#file_path=raw_input("Please enter the directory of reuters dataset: ") 
file_path = "./data/"

#iterate through all files in the given directory
for file in listdir(file_path):
    soup = BeautifulSoup(open(file_path+file),"html.parser")
    Document_List += get_word_frequency(soup)
    break
#remove words only show up once among all documents  
for key in Vocabulary.keys():
    if Vocabulary[key] == 1:
	del Vocabulary[key]

import math

sorted_Vocabulary = sorted(Vocabulary.items(), key=operator.itemgetter(0))
idf = []
for idx, item in enumerate(sorted_Vocabulary):
	print (item[0] + " in " + str(item[1]) + " documents")
	Vocabulary[item[0]] = idx
	idf.append(math.log(float(len(Document_List))/float(item[1])))

print ("Vacabulary has %d words." % len(Vocabulary))

import numpy as np
from scipy.sparse import csr_matrix

tf_idf_matrix = csr_matrix((len(Document_List), len(sorted_Vocabulary)), dtype=float)

for idx, D in enumerate(Document_List):
	print ("Caculating vector for document #" + str(D.id))
	vector = len(sorted_Vocabulary)*[0.0]
	max_freq = 0
	for key in D.freq:
		if key in Vocabulary:
			max_freq = max_freq if max_freq >= D.freq[key] else D.freq[key]
	for key in D.freq:
		if key in Vocabulary:
 			tf = float(D.freq[key])/float(max_freq)
			word_id = Vocabulary[key]
			tf_idf_matrix[idx][word_id] = tf * idf[word_id]

	#construct feature vector using word frequency
	#D.freq_vector = vector
	D.freq = {}
	
	#compute tf-idf
	#max_freq = max(vector)
	#if max_freq == 0:
	#	max_freq = 1
	#for f in vector:
	#	f = (f/max_freq)
	#	tf_idf_matrix[idx][]
	#	tf_idf.append(math.log(len(Document_List)/f))
	#tf_idf = map(operator.mul, vector, tf_idf)
	#construct feature vector using tf-idf
	#D.tf_idf_vector = tf_idf 

import pickle

with open('document_data.pkl', 'w') as output:
	for D in Document_List:
		pickle.dump(D, output, 1)

with open('vocabulary.pkl', 'w') as output:
	pickle.dump(sorted_Vocabulary, output, 0)


