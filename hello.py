#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import Flask


# In[6]:


app = Flask(__name__)

@app.route('/')
def helloworld():
    return('Hello World!!')

