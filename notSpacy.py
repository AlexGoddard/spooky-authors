import spacy, re
import pandas as pd

nlp = spacy.load('en')
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
postagfile = open('./postag.txt', 'w')
wantedTags = re.compile(r'(\w+)\,\s?u\'NN.?\'|(\w+)\,\s?u\'JJ\'|(\w+)\,\s?u\'VB.\'')
tempRegex = re.compile(r'\w+')

jjregex = re.compile(r'(\w+)\,\s?u\'JJ\'')
nnregex = re.compile(r'(\w+)\,\s?u\'NN.?\'')
vbregex = re.compile(r'(\w+)\,\s?u\'VB.\'')

EAPdict = list()
EAPtags = list()
MWSsentences = list()
HPLsentences = list()

def displayPandas(content):
	return content.head()


def getData():
	for index, row in train.iterrows():
		if row['author'] == 'EAP':
			EAPdict.append(row['text'])
			# print(row['text'], row['author']
		elif row['author'] == 'MWS':
			MWSsentences.append(row['text'])
		else:
			HPLsentences.append(row['text'])

def findSimilarity(sent1, sent2):
	u1 = str(sampleSentence1)
	u1 = nlp(u1)
	u2 = str(sampleSentence2)
	u2 = nlp(u2)
	return u1.similarity(u2)

def print_fine_pos(token):
    return (token.tag_)

def pos_tags(sentence):
    tokens = nlp(sentence)
    tags = []
    for tok in tokens:
        tags.append((tok,print_fine_pos(tok)))
    return tags

def parsePOS(annotatedSentence):
	words = list()
	matches = re.findall(wantedTags, str(annotatedSentence))
	for match in matches:
		print(tempRegex.findall(str(match)))
		words.append(tempRegex.findall(str(match)))
	return words

def buildWords(sentences, author):

	if author == 'EAP':
		eapauthorwords = open('./eap.txt', 'w')
		for sentence in sentences:
			eapauthorwords.write(str(parsePOS(pos_tags(sentence))))
			eapauthorwords.write('\n')
	elif author == 'HPL':
		hplwordsfile = open('./hpl.txt', 'w')
		for sentence in sentences:
			hplwordsfile.write(str(parsePOS(pos_tags(sentence))))
			hplwordsfile.write('\n')
	else:
		mwsfile = open('./mws.txt', 'w')
		for sentence in sentences:
			mwsfile.write(str(parsePOS(pos_tags(sentence))))
			mwsfile.write('\n')


def countPOStagsConsolidated(annotatedSentence):
	matches = wantedTags.findall(str(annotatedSentence))
	count = 0
	for match in matches:
		print(tempRegex.findall(str(match)))
		count = count + 1
	print(count)

def countPosTagsIndividually(annotatedSentence):

	nnmatches = nnregex.findall(str(annotatedSentence))
	jjmatches = jjregex.findall(str(annotatedSentence))
	vbmatches = vbregex.findall(str(annotatedSentence))

	return len(nnmatches), len(jjmatches), len(vbmatches)


getData()

'''
Testing each module below
'''
print("EAP Sentences: {}, MWS Sentences: {}, HPL Sentences: {}".format(len(EAPdict), len(MWSsentences), len(HPLsentences)))
sampleSentence1 = EAPdict[0]
sampleSentence2 = HPLsentences[3]

print("Sentence 1 pos tag -> {}".format(pos_tags(sampleSentence1)))
print("Sentence 2 pos tag -> {}".format(pos_tags(sampleSentence2)))
print("Score for sentence 2 -> {}".format(parsePOS(pos_tags(sampleSentence1))))

'''
Calling the counter for counting how many syntactic dependencies in a sentence
'''

print("Nouns, adjectives, verbs = {}".format(countPosTagsIndividually(pos_tags(sampleSentence1))))



