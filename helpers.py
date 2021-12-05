import re
import pickle

def removePunctuation(text):
    text=re.sub(r'[^A-za-z\s]','',text)#remove puntuations
    text=re.sub(r'\S*\d\S*','',text)#remove alphanumeric words
    return text

#remove html tags
def removeHTML(text):
    text=re.sub(r'https?\S+','',text)
    text=re.sub(r'<.*?>','',text)
    return text

#remove unwanted text from the reviews like instead of i've keep i have
#replace n't, 'll, 's, 've, 're, 't, 'd, 'm
#from tqdm.notebook import tqdm
def subsitute(text):
    text=re.sub(r'can\'t','can not',text)
    text=re.sub(r'won\'t','will not',text)

    text=re.sub(r'n\'t',' not',text)
    text=re.sub(r'\'ll',' will',text)
    text=re.sub(r'\'s',' is',text)
    text=re.sub(r'\'ve',' have',text)
    text=re.sub(r'\'re',' are',text)
    text=re.sub(r'\'t',' not',text)
    text=re.sub(r'\'d',' would',text)
    text=re.sub(r'\'m',' am',text)
    return text

with open('LR_model.pkl','rb') as file:
    clf = pickle.load(file)

with open('tfidf_vectorizer.pkl','rb') as file:
    tfidf_vectorizer = pickle.load(file)
