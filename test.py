"""
Lab1 data preprocessing

@author: Zhe Dong, Kun Liu
"""

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
                    for topic in doc.topics.children:                    
                        D.topics.append(unicode(doc.find("topics").string))
                if doc.find("places") is not None:
                    for place in doc.places.children:                    
                        D.places.append(unicode(doc.find("places").string))
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
				if len(word)<2:
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
    
 #remove words only show up once among all documents  
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
	#construct feature vector using word frequency
	D.freq_vector = vector
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
	#construct feature vector using tf-idf
	D.tf_idf_vector = tf_idf 

import pickle

with open('document_data.pkl', 'w') as output:
	for D in Document_List:
		pickle.dump(D, output, 0)

with open('vocabulary.pkl', 'w') as output:
	pickle.dump(sorted_Vocabulary, output, 0)


