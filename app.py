#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from nltk.stem.snowball import SnowballStemmer
#import pickle
#import re
from helpers import clf, tfidf_vectorizer, removePunctuation, removeHTML, subsitute
from flask import Flask, render_template, request


# In[ ]:
'''

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
'''
stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',             's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",             'won', "won't", 'wouldn', "wouldn't"])

snbstemmer = SnowballStemmer('english')
'''
with open('LR_model.pkl','rb') as file:
    clf = pickle.load(file)

with open('tfidf_vectorizer.pkl','rb') as file:
    tfidf_vectorizer = pickle.load(file)
'''

# In[ ]:

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/result',methods=['POST'])
def result():
    review = request.form['review']
    review= removeHTML(review)#removes html tags and attributes
    review= subsitute(review)#converts short words to normal words
    review= removePunctuation(review).strip()#removes punctuation and alphanumeric words
    review = ' '.join(ele for ele in review.split() if ele not in stopwords)
    words = review.split()
    review = ' '.join([snbstemmer.stem(w) for w in words])
    xtest = tfidf_vectorizer.transform([review])
    result = clf.predict(xtest)
    res_prob = clf.predict_proba(xtest)
    if result:
        s = "The review is Positive with probability: {:.2f}".format(res_prob[0][1])
    else:
        s = "The review is Negative with probability: {:.2f}".format(res_prob[0][0])
    return render_template('result.html',to_send=s)


if __name__=='__main__':
    app.run(host='0.0.0.0', port='5005')
