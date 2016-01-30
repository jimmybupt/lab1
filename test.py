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
from os import listdir
from bs4 import BeautifulSoup

stemmer = nltk.stem.porter.PorterStemmer() 
stop = nltk.corpus.stopwords.words("english")
Vocabulary = {}
Document_List = []
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
                D = Document()
		id = doc.get("newid")
                print "parsing document #" + id
                D.title = doc.find("title").string if doc.find("title") is not None else ""
		D.topic = doc.find("topic").string if doc.find("topic") is not None else ""
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
    Document_List.extend(get_word_frequency(soup))
    break
    
for key in Vocabulary.keys():
    if Vocabulary[key] == 1:
	del Vocabulary[key]
    else:
	print key + " in " + str(Vocabulary[key]) + " documents"

print "Vacabulary has %d words." % len(Vocabulary)

for D in Document_List:
	vector = []
	#print freq
	for key in Vocabulary:
		if key in D.freq:
			vector.append(D.freq[key])
		else:
			vector.append(0)
	D.vector = vector
	#print vector


