#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from tqdm.notebook import tqdm
sns.set()


# In[2]:


#create connection to sql database
conn = sqlite3.connect('database.sqlite')


# In[3]:


q = """SELECT name FROM sqlite_master WHERE type='table'"""
tables = pd.read_sql_query(q,conn)


# In[4]:


tables


# In[5]:


q = """SELECT * FROM Reviews WHERE Score != 3"""
df = pd.read_sql_query(q,conn)
df.head()


# In[6]:


#568454
df.info()


# 1. Id - the id of the row
# 2. ProductId- the id of the particular product
# 3. UserId - the id of the user 
# 4. ProfileName - the profile name of the user
# 5. HelpfulnessNumerator - the no of user who found the review helpful
# 6. HelpfulnessDenominator - the no of user who voted the answer
# 7. Score - the rating given by the user to the product
# 8. Time - the time when the review was added
# 9. Summary -  the title of the review
# 10. Text -  the actual review

# In[7]:


#Data cleaning
#check for reviews other than food
#check for duplicate entries
#check for inconsistent data
df.sort_values('ProductId',axis=0,inplace=True)
clean_df = df.drop_duplicates(subset={'UserId','Score','Time','Text'},keep='first')
clean_df.shape


# In[8]:


#remaining %of data
sum(clean_df['Id'].value_counts())/sum(df['Id'].value_counts())


# In[9]:


#inconsistent data
clean_df.loc[clean_df.HelpfulnessNumerator>clean_df.HelpfulnessDenominator]


# In[10]:


#drop inconsistent data
clean_df1 = clean_df.drop(clean_df['Id'].loc[clean_df.HelpfulnessNumerator>clean_df.HelpfulnessDenominator])


# In[11]:


#convert the Text and Summary cols into lower case
lower_summary = clean_df1.Summary.str.lower()
clean_df1['Summary'] = lower_summary
lower_text = clean_df1.Text.str.lower()
clean_df1['Text'] = lower_text


# In[12]:


clean_df1.reset_index()
clean_df1.shape


# In[13]:


import re
#removing reviews other than food like book or music
patterns=[r'\bbooks?\b','\breads?\b',r'\breading\b',r'\bpoetry\b',r'\bmusic\b',r'\bplay\b',r'\bplaying\b',r'\bmovies?\b',r'\bpoems?\b']
final=clean_df1
for pattern in tqdm(patterns):
    final=final.drop(list(final[final.Text.str.contains(pattern)].index),axis=0)
    final=final.drop(list(final[final.Summary.str.contains(pattern)].index),axis=0)
final


# In[14]:


#remaining data size
sum(final['Id'].value_counts())/sum(df['Id'].value_counts())


# In[15]:


#remove unwanted text from the reviews like instead of i've keep i have 
#replace n't, 'll, 's, 've, 're, 't, 'd, 'm
from tqdm.notebook import tqdm
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
    


# In[16]:


#remove html tags
def removeHTML(text):
    text=re.sub(r'https?\S+','',text)
    text=re.sub(r'<.*?>','',text)
    return text


# In[17]:


def removePunctuation(text):
    text=re.sub(r'[^A-za-z\s]','',text)#remove puntuations
    text=re.sub(r'\S*\d\S*','',text)#remove alphanumeric words
    return text


# In[18]:


stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',             's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",             'won', "won't", 'wouldn', "wouldn't"])


# In[19]:


#clean the text review column and store in a list
final_reviews=[]
#test=[""""newman's own" products are generally very good, with high quality ingediants and perfect execution.  this organic dark chocolate bar (54% cocoa) was perfectly sweet, without being bitter or too rich.  two of us spent an hour slowly munching through one bar, savoring every bite.<br /><br />a real treat of high quality dark chocolate.  definitely worth a try!"""]
#final.Text.values
for text in tqdm(final.Text.values):
    text= removeHTML(text)#removes html tags and attributes
    text= subsitute(text)#converts short words to normal words
    text= removePunctuation(text).strip()#removes punctuation and alphanumeric words
    text = ' '.join(ele for ele in text.split() if ele not in stopwords)
    final_reviews.append(text.strip())
final_reviews


# In[20]:


#clean the summary column and store in a list
final_summary=[]
#test=[""""newman's own" products are generally very good, with high quality ingediants and perfect execution.  this organic dark chocolate bar (54% cocoa) was perfectly sweet, without being bitter or too rich.  two of us spent an hour slowly munching through one bar, savoring every bite.<br /><br />a real treat of high quality dark chocolate.  definitely worth a try!"""]
#final.Text.values
for text in tqdm(final.Summary.values):
    text= removeHTML(text)#removes html tags and attributes
    text= subsitute(text)#converts short words to normal words
    text= removePunctuation(text).strip()#removes punctuation and alphanumeric words
    text = ' '.join(ele for ele in text.split() if ele not in stopwords)
    final_summary.append(text.strip())
final_summary


# In[21]:


#generate random reviews to check
for i in range(10):
    r=np.random.randint(0,356001)
    print(final.Text.iloc[r])
    print('='*100)


# In[22]:


#Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction import stop_words


# In[23]:


snbstemmer = SnowballStemmer("english")
wnl = WordNetLemmatizer()


# In[24]:


summary_stem=[]
for ele in final_summary:
    words = ele.split()
    summary_stem.append(' '.join([snbstemmer.stem(word) for word in words]))


# In[25]:


summary_stem[0:10]


# In[26]:


summary_lemma=[]
for ele in final_summary:
    words = ele.split()
    summary_lemma.append(' '.join([wnl.lemmatize(word) for word in words]))


# In[27]:


summary_lemma[0:10]


# In[28]:


text_stem=[]
for ele in final_reviews:
    words = ele.split()
    text_stem.append(' '.join([snbstemmer.stem(word) for word in words]))


# In[29]:


text_lemma=[]
for ele in final_reviews:
    words = ele.split()
    text_lemma.append(' '.join([wnl.lemmatize(word) for word in words]))


# In[30]:


text_stem[0:10]


# In[31]:


type(text_stem)


# In[32]:


text_lemma[0:10]


# In[33]:


#now we have reduced the words into their stem/lemma forms
#using this we can create BoW, TF-IDF or W2Vec to actually convert the words into numbers\
#which we can use then for training classification algorithms.
#we will go with stemming
bow_summary = CountVectorizer(ngram_range=(1,2),min_df=10,max_features=5000)
summary_final_bow_vec = bow_summary.fit_transform(summary_stem)
bow_text = CountVectorizer(ngram_range=(1,2),min_df=10,max_features=5000)
text_final_bow_vec = bow_text.fit_transform(text_stem)


# In[34]:


bow_summary.get_feature_names()[:100] #viewing 100 features of summary out of 5000


# In[35]:


bow_text.get_feature_names()[4900:] #viewing 100 features of reviews


# In[36]:


tfidf_summary = TfidfVectorizer(ngram_range=(1,2),min_df=10,max_features=5000)
summary_tfidf= tfidf_summary.fit_transform(final_summary)
tfidf_text = TfidfVectorizer(ngram_range=(1,2),min_df=10,max_features=5000,lowercase=False)
text_tfidf = tfidf_text.fit_transform(text_stem)


# In[37]:


tfidf_summary.get_feature_names()#top 100 features of summary using tfidf


# In[38]:


tfidf_text.get_feature_names()#top 100 features of text using tfidf


# In[39]:


summary_final_bow_vec.get_shape()


# In[40]:


text_final_bow_vec.get_shape()


# In[41]:


summary_tfidf.get_shape()


# In[42]:


y=final['Score']
y.columns=['idx','Score']
y = pd.DataFrame(y)
y


# In[43]:


y.Score.loc[y.Score<3]=0
y.Score.loc[y.Score>3]=1
y


# In[44]:


# Train test split
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(text_tfidf,y,train_size=0.7,random_state=42)


# In[45]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score as score
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

params = {'alpha':range(1,15)}
f1score = make_scorer(score)

clf = MultinomialNB()
clf2 = RandomizedSearchCV(clf,params,scoring=f1score,verbose=5,random_state=42)
#clf.fit(text_tfidf,y)


# In[46]:


clf2.fit(xtrain,ytrain)


# In[47]:


clf2.best_params_


# In[48]:


# Texting Phase
# Clean the review text by removing stop words, punctuations etc.
text="""bad taste. won't reccomend this to anyone"""
text= removeHTML(text)#removes html tags and attributes
text= subsitute(text)#converts short words to normal words
text= removePunctuation(text).strip()#removes punctuation and alphanumeric words
text = ' '.join(ele for ele in text.split() if ele not in stopwords)
test_review=text.strip()
test_review


# In[49]:


# Converting the review to its stem form
words = test_review.split()
test_review = ' '.join([snbstemmer.stem(w) for w in words])
test_review


# In[50]:


test = [test_review]
test= tfidf_text.transform(test)
test


# In[70]:


a=clf4.predict_proba(test)
a[0][1]


# In[73]:


s = "{:.2f}".format(a[][0])
s


# In[52]:


yhat = clf2.predict(xtest)


# In[53]:


result = pd.DataFrame(yhat)
result


# In[54]:


score(ytest,yhat)


# In[55]:


from sklearn.linear_model import LogisticRegression

clf3= LogisticRegression()
params={'C':[3,4,5,6,7,8]}
clf4 = RandomizedSearchCV(clf3,params,scoring=f1score,random_state=42,verbose=10)


# In[56]:


clf4.fit(xtrain,ytrain)


# In[57]:


clf4.best_estimator_


# In[58]:


yhat_logit= clf4.predict(xtest)


# In[59]:


score(ytest,yhat_logit)


# In[62]:


clf4 = LogisticRegression(C=3)
clf4.fit(xtrain,ytrain)


# In[63]:


import pickle

tfidf_file = 'tfidf_vectorizer.pkl'
lr_file = 'LR_model.pkl'

with open(tfidf_file,'wb') as file:
    pickle.dump(tfidf_text,file)
    
with open(lr_file,'wb') as file:
    pickle.dump(clf4,file)


# In[ ]:


clf4.predict()

