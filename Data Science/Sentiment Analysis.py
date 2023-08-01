#!/usr/bin/env python
# coding: utf-8

# In[1]:


from textblob import TextBlob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


reviews = pd.read_csv("Restaurant_Reviews.csv")
reviews.sample(3)


# In[9]:


reviews.shape


# In[10]:


reviews["Sentiment"] = "Sentiment"
reviews.sample(3)


# In[11]:


# Function to perform sentiment analysis using TextBlob
def analyze_sentiment(review):
    blob = TextBlob(review)
    sentiment_score = blob.sentiment.polarity

    # Sentiment polarity ranges from -1 to 1.
    # If the sentiment score is greater than 0, the review is positive.
    # If the sentiment score is less than 0, the review is negative.
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"


# In[12]:


# Perform sentiment analysis on each review
for i in range(0, reviews.shape[0]):
    review = reviews.iloc[i][0]
    sentiment = analyze_sentiment(review)
    reviews.Sentiment.iloc[i] = reviews.Sentiment.iloc[i].replace("Sentiment", sentiment)


# In[13]:


reviews.sample(5)


# In[17]:


sentiment_counts = reviews.Sentiment.value_counts()

plt.pie(sentiment_counts, 
        labels=sentiment_counts.index,
        autopct='%1.1f%%')

plt.title("Sentiment Analysis")
plt.legend(labels=sentiment_counts.index)
plt.axis('equal') 

plt.show()


# In[ ]:




