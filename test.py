#Lab1

#Another thing we forgot to add is the class labels as provided in the TOPICS and PLACES tabs of each article

class Document:
	def __init__(self):
		self.title = ""
		self.topic = ""
		self.id = 0
		self.freq = {}
		self.vector = []


import nltk
import copy
import sys
sys.setrecursionlimit(100000)
import operator
from os import listdir
from bs4 import BeautifulSoup

stemmer = nltk.stem.porter.PorterStemmer() 
stop = nltk.corpus.stopwords.words("english")
Vocabulary = {}
Document_List = []

def get_word_frequency(soup):
    currentFreq=[]
    for doc in soup.find_all("reuters"):
                D = Document()
		id = doc.get("newid")
                print "parsing document #" + id
                if doc.find("title") is not None:
			D.title = unicode(doc.find("title").string)
			print D.title
		if doc.find("topics") is not None:
			D.topic = unicode(doc.find("topics").string)
		D.id = int(id)
		D.freq = {}
                if doc.find("body") is not None:
                        words = nltk.word_tokenize(doc.find("body").string)
                        for word in words:
                                word = word.lower()
				if not word[0].isalpha():
					continue
				if word in stop:
					continue
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
    
for key in Vocabulary.keys():
    if Vocabulary[key] == 1:
	del Vocabulary[key]

sorted_Vocabulary = sorted(Vocabulary.items(), key=operator.itemgetter(0))
for item in sorted_Vocabulary:
	print item[0] + " in " + str(item[1]) + " documents"

print "Vacabulary has %d words." % len(Vocabulary)

import math
for D in Document_List:
	vector = []
	doc_count = []
	tf_idf = []
	for item in sorted_Vocabulary:
		doc_count.append(float(item[1]))
		if item[0] in D.freq:
			vector.append(float(D.freq[item[0]]))
		else:
			vector.append(0.0)
	#D.vector = vector
	D.freq = {}
	
	#compute tf-idf
	max_freq = max(vector)
	if max_freq == 0:
		max_freq = 1
	for f in vector:
		f = 0.5 + 0.5 * (f/max_freq)
	for f in doc_count:
		tf_idf.append(math.log(len(Document_List)/f))
	tf_idf = map(operator.mul, vector, tf_idf)
	D.vector = tf_idf 

import pickle

with open('document_data.pkl', 'w') as output:
	for D in Document_List:
		pickle.dump(D, output, 0)

with open('vocabulary.pkl', 'w') as output:
	pickle.dump(sorted_Vocabulary, output, 0)


