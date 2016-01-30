#Lab1

#Another thing we forgot to add is the class labels as provided in the TOPICS and PLACES tabs of each article

import nltk
from os import listdir
from bs4 import BeautifulSoup

stemmer = nltk.stem.porter.PorterStemmer() 
stop = nltk.corpus.stopwords.words("english")
Vocabulary = {}
Freq = []
#print soup.prettify()

def is_number(str):
	try:
        	float(str)
        	return True
    	except ValueError:
        	return False

def get_word_frequency(soup):
    currentFreq=[]
    for doc in soup.find_all("reuters"):
                id = doc.get("newid")
                print "parsing document #" + id
                freq = {}
                if doc.find("body") is not None:
                        words = nltk.word_tokenize(doc.find("body").string)
                        for word in words:
                                word = word.lower()
				if is_number(word):
					continue
				if word in stop:
					continue
                                word = stemmer.stem(word) #get words stemmed
                                if word not in freq:
                                        if word not in Vocabulary:
                                                Vocabulary[word] = 1
                                        else:
                                                Vocabulary[word] += 1
                                        freq[word] = 1
                                else:
                                        freq[word] += 1
    currentFreq.append(freq)
    return currentFreq
    
#ask for directory for dataset
#file_path=raw_input("Please enter the directory of reuters dataset: ") 
file_path = "./data/"

#iterate through all files in the given directory
for file in listdir(file_path):
    soup = BeautifulSoup(open(file_path+file),"html.parser")
    Freq.extend(get_word_frequency(soup))
    break
    
for key in Vocabulary.keys():
    if Vocabulary[key] == 1:
	del Vocabulary[key]
    else:
	print key + " in " + str(Vocabulary[key]) + " documents"

print "Vacabulary has %d words." % len(Vocabulary)

for freq in Freq:
	vector = []
	#print freq
	for key in Vocabulary:
		if key in freq:
			vector.append(freq[key])
		else:
			vector.append(0)
	#print vector


