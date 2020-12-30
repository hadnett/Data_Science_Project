#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: williamhadnett D00223305
"""

# =============================================================================
# STEP 2 - Data Mining
# =============================================================================

from bs4 import BeautifulSoup
import requests
import time
import pandas as pd
import os
import numpy as np
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from nltk.corpus import stopwords
import re
from wordcloud import WordCloud


os.chdir('/Users/williamhadnett/Documents/Data_Science/Data_Science_Project_William_Hadnett')

# Discovered a way to find elements in the HTML tree via Beautiful Soup's 
# find() function https://www.crummy.com/software/BeautifulSoup/bs4/doc/#find.
# The find_all function works in a similiar manner. However, it finds every
# occurrence of an element in a given HTML page and as the posts are displayed
# via the li tag this function can retrieve all posts from the html page. 

# The class name of the element can also be given to this find() function to target
# specific HTML elements. https://www.crummy.com/software/BeautifulSoup/bs4/doc/#attrs

# I decided to use this approach as it simplifies the process of parsing the HTML
# while also reducing code complexity and making the code more readable.

# While mining the data for this project I discovered two primary issues. The first
# was whether the news was true or false as the website being scraped stores this
# information in the form of an image. To overcome this issue I simply manipulated the
# ruling in the url of the website by passing the kind of posts I wanted returned 
# (true or false) as an agruement to the function. I then stored this function
# agruement in the dataframe. The second issue was the date of the post. In the 
# HTML it is displayed as a string with text surrounding it. I have made the decision
# to scrape this entire string and then process this string for the date in the data
# cleaning section of this project.  

post_details = []
scrapeData = False

def parseHTML(isTrue):
    for i in range(1,25,1):
        url= 'https://www.politifact.com/factchecks/list/?page='+str(i)+'&ruling='+ isTrue +''
        html = requests.get(url).text
        soup = BeautifulSoup(html,	'html5lib')
        soup = soup.find_all('li',class_ = 'o-listicle__item')
        for post in soup:
            quote = post.find("div", class_ = 'm-statement__quote').text.strip()
            source = post.find("a", class_ = 'm-statement__name').text.strip()
            date = post.find("div", class_ = 'm-statement__desc').text.strip()
            post_details.append([source, quote, date, isTrue])
        time.sleep(2)

if scrapeData:
    # Get True News Stories
    parseHTML('true')
    
    # Get Fake News Stories
    parseHTML('false')
    
    # 1440 posts retrieved.
    
    df = pd.DataFrame(post_details, columns=['Source', 'Quote', 'Date', 'isTrue'])
    
    # Write data to CSV file for future use. 
    df.to_csv('fake_news.csv', encoding="utf-8", index=False)
else:
    df = pd.read_csv('fake_news.csv')

# =============================================================================
# STEP 2 - Data Mining - Specify data characteristics. 
# =============================================================================

df.info()

#  #   Column  Non-Null Count  Dtype 
# ---  ------  --------------  ----- 
# 0   Source  1440 non-null   object
# 1   Quote   1440 non-null   object
# 2   Date    1440 non-null   object
# 3   isTrue  1440 non-null   bool  
# dtypes: bool(1), object(3)

# Currently the 'Source', 'Quote' and 'Date' are all strings. In the data 
# cleaning section of this file I intend to convert the 'Date' from a string
# to a Date Object for further processing. The response variable 'isTrue' is 
# represented as a boolean. 

# Check first 5 rows in data to ensure the data collected looks
# correct.
df.head()

describe = df.describe()

#                 Source  ... isTrue
# count             1440  ...   1440
# unique             524  ...      2
# top     Facebook posts  ...   True
# freq               302  ...    720

# While some of the data is omitted in the terminal if we view the variable
# 'describe' in the Variable Explorer window we can see that there are 1440
# records in total. There are 524 unique soruces, 1439 unique quotes, 1234 
# unique dates and 2 unique response variables and this all appears to be 
# reasonable. The top source was 'Facebook Posts' appearing 302 times, the
# top quote appeared 2 times and the top date was the 4th November 2020. 
# Note 'top' is similiar to the mode as it represents the data which appears
# most often in a column.

# =============================================================================
# STEP 2 - Data Mining - Identify Variables  
# =============================================================================

# isTrue - Response Variable - Categorical - Classification Model Required
# Source - Predictor Variable - Categorical
# Quote - Predictor Variable - Categorical
# Date - Predictor Variable - Can be both numerical and categorical
# TODO review date and how it has been used.

# While there are currently only 3 predictor variables I intend to introduce
# more variables to help with this analysis during the Feature Engineering 
# stage of this project. I believe that variables such as gender, sentiment,
# length of text and most frequent words used for both fake and real news 
# may help refine this classification model.   

# =============================================================================
# STEP 3 - Data Cleaning
# =============================================================================

# Remove quotation marks from quotes in dataframe.

# Discovered that two different types of quotes where present within this dataset
# removed both sets to avoid issues with processing later during this project.

df['Quote'] = df['Quote'].str.strip('[",“,”]') 
df['Quote'] = df['Quote'].str.replace('[",“,”]', '')

# Parse date field to determine date of post.
# Started by separating the date data found into three separate columns.
# Found this approach useful as it can be used for categorical analysis later
# to determine quantity of fake news by month or year and this approach also 
# simplifed the conversion to datetime using a pandas dataframe.
df['Month'] = df['Date'].str.split().str[2]
df['Day'] = df['Date'].str.split().str[3]
df['Year'] = df['Date'].str.split().str[4]

# Cleaned up the date data as whitespace and other irrelevant characters (':',',') 
# where present after seperating the date data into three different columns.
df['Month'] = df['Month'].str.strip()

# This approach was discovered after I tried to convert the long month name 
# 'June' to a date. Pandas to_datetime function cannot parse long month names.
# Therefore, to overcome this problem the month name as been converted to an int.
# Converting Month to Number: https://stackoverflow.com/questions/48122046/pandas-convert-month-name-to-int-concat-to-column-and-convert-to-date-time
df['Month'] = pd.to_datetime(df.Month, format='%B').dt.month

df['Day'] = df['Day'].str.strip(',')
df['Year'] = df['Year'].str.strip()
df['Year'] = df['Year'].str.strip(':')

# Convert to datetime, but only store date as time is irrelevant for this 
# analysis.
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']]).dt.date

# Check for null values in the dataset.
df.isnull().sum()

# Source    0
# Quote     0
# Date      0
# isTrue    0
# Month     0
# Day       0
# Year      0
# No Null/Nan values present in the dataset.

# We know from step 2 that there is a duplicate quote in the dataset. I believe
# this duplicate should be removed. In this dataset the occurence of one duplicate
# will probably not hinder the model produced greatly. However, if this dataset 
# grows and a number of duplicates increase then this could cause a bias that 
# may lead to the overfitting of this model. Producing a model that performs 
# well on the training data but poorly on the test data. 
describe = df['Quote'].describe()
df = df.drop_duplicates(subset='Quote', keep='first')
# No further duplicate quotes.
describe = df['Quote'].describe()


# =============================================================================
# STEP 3 - Data Exploration - Univariate
# =============================================================================

sourceCount = df.Source.value_counts()

# Bar chart showing the top 10 sources. 
sourceCount.plot.bar()
plt.title("Top 10 Sources")
plt.ylabel("Total Quotes")
plt.xlabel("Source")
plt.xlim(0,10)
plt.show()

# It is clear from the bar chart displayed above that the top sources are primarily
# social media platforms and politicians. The worlds top 3 social media platforms 
# Facebook, Twitter and Instagram are all present within the top 10 sources with Facebook
# ranking as the top source with 302 quotes. Other miscellaneous sources are also present
# within the top ten such as 'Viral Image' and 'Bloggers'. I do not think it is unreasonable
# to assume that both 'Viral Images' and 'Bloggers' also used social media platforms to 
# distrubute their content. Some of the worlds most famous politicians are also present
# Donald Trump (97 Quotes), Joe Biden (24 Quotes) and Hillary Clinton (38 Quotes) 
# are also present.  

dateCount = df.Date.value_counts()

# Bar chart showing the 10 Most Popular Dates for posting.
dateCount.plot.bar()
plt.title("10 Most Popular Dates")
plt.ylabel("Total Quotes")
plt.xlabel("Date of Quote")
plt.xlim(0,10)
plt.show()

# 04-11-2020 was the most popular date for quotes posted with a total of
# 21 quotes posted on that day. This makes sense as we know the US elections
# took place on the 03-11-2020. The 05-11-2020 was the second busiest day with
# 14 quotes. The number of quotes posted per day then begins to level out 
# ranging from 10 posts per day to 1 post per day.

# Check split of true vs false news.
numberTrueFalse = df.isTrue.value_counts(normalize=True)
numberTrueFalse.plot.pie()
plt.show()
# Split: False = 0.500347, True = 0.499653
# Balanced data is important for generating a classification model according to:
# https://www.r-bloggers.com/2020/06/why-balancing-your-data-set-is-important/#:~:text=From%20the%20above%20examples%2C%20we,set%20for%20a%20classification%20model.
# It is clear that the data selected for this project has a good balance of 
# reliable and unreliable news quotes. 

# Find max and min date: 
dateMin = df.Date.min()
# dateMin = 2012-05-22
dateMax = df.Date.max()
# dateMax = 2020-12-22

# Get Average Length of Quote
averageLength = df['Quote'].apply(len).mean()
# averageLength = 102.33912439193885

# Shortest Quote: 
shortestQuote = df['Quote'].apply(len).min()
# shortestQuote = 23

# Longest Quote:
longestQuote = df['Quote'].apply(len).max()
# longestQuote = 313 

# Top 15 most commons words used in quotes:

# Multitude of methods for counting frequency of words in a pandas dataframe.
# Inititally implemented a Counter from the collections library. However, when 
# ran it returned words such as 'says', 'I' and 'me' (words with little value).
# Discovered that nltk offers a collection of these 'stopwords' which can be used
# to exlucde these irrelevant words from the count. The code below was researched 
# via Stack Overflow. However, it has been changed as I currently do not want to 
# store these frequently appearing words in the dataframe. 
# https://stackoverflow.com/a/56134977/11058925
# Words such as 'says' also appear frequently in this dataset and are not included in
# the collection of stopwords. Therefore, I added it to the stopwords collection
# as it adds no value to this analysis.

stop = stopwords.words('english') 
newStopWords = ['says', 'one', 'people', 'said', 'since', 'new', 'shows', 'would']
stop.extend(newStopWords)

frequentWords = df['Quote'].str.lower() \
        .apply(lambda x: 
           ' '.join([word for word in str(x).split() if word not in (stop)])) \
        .str.split(expand=True).stack().value_counts()

frequentWords.plot.bar()
plt.title("Top 15 Words")
plt.ylabel("Total Appearances")
plt.xlabel("Word")
plt.xlim(0,15)
plt.show()

# We can see from the bar chart displayed above that the majority of the most 
# frequently used words relate to US politicians. Trump appears 120 times 
# throughout the dataset with Donald appearing 89 times. Surprisingly
# COVID-19 only appeared 49 times. Currently this analysis of the most 
# frequently used words does not offer much insight into the business problem.
# However, in the next section I intend to review which of these words are 
# assosicated with reliable and unreliable news quotes. 

# =============================================================================
# STEP 3 - Data Exploration - Bivariate
# =============================================================================
# df[df.a > 1].sum() 
sourceTrue = df.groupby('Source')['isTrue'].apply(lambda x: x[x == True].count())
sourceTrue = sourceTrue.sort_values(ascending=False)

sourceTrue.plot.bar()
plt.title("Top 10 Reliable Sources")
plt.ylabel("Total True Posts")
plt.xlabel("Source")
plt.xlim(0,10)
plt.show()

sourceFalse = df.groupby('Source')['isTrue'].apply(lambda x: x[x == False].count())
sourceFalse = sourceFalse.sort_values(ascending=False)

sourceFalse.plot.bar()
plt.title("Top 10 Unreliable Sources")
plt.ylabel("Total False Posts")
plt.xlabel("Source")
plt.xlim(0,10)
plt.show()

# Upon reviewing both bar charts generated above there are a number of interesting
# points. Firstly, Donald Trump appears in both categories along with Joe Biden. 
# However, the top 10 reliable sources are almost all politicians/leaders/individuals. 
# Whereas, 7 out of top 10 unreliable sources are social media platforms (Twitter,
# Facebook, YouTube, Instagram, etc) this does not mean that these platforms are
# unreliable sources but they do facility the spreading of fake news. This discovery
# would lead me to believe that the majority of fake news is spread via one of these 
# platforms. 

statsReliability = df.groupby('isTrue')['Quote'].apply(lambda x: x[x.str.contains('\\d', regex=True)].count())

# False    224
# True     310

statsReliability.plot.pie()
plt.title('Quotes Containing Digits')
plt.show()

# It appears that news quotes that contain digits (aka stats) are more likely to
# to be True. This is surprising. I thought that unreliable quotes
# would have used stats to leverage peoples opinions. 

# Most Frequently used words in reliable and unreliable news quotes.

reliableData = df[df["isTrue"] == True]

frequentWordsReliable = reliableData.Quote.str.lower() \
        .apply(lambda x: 
           ' '.join([word for word in str(x).split() if word not in (stop)])) \
        .str.split(expand=True).stack().value_counts()
        
frequentWordsReliable.plot.bar()
plt.title("10 Most Frequent Words in Reliable News")
plt.ylabel("Count")
plt.xlabel("Word")
plt.xlim(0,10)
plt.show()
        
unreliableData = df[df["isTrue"] == False]

frequentWordsUnreliable = unreliableData.Quote.str.lower() \
        .apply(lambda x: 
           ' '.join([word for word in str(x).split() if word not in (stop)])) \
        .str.split(expand=True).stack().value_counts()

frequentWordsUnreliable.plot.bar()
plt.title("10 Most Frequent Words in Unreliable News")
plt.ylabel("Count")
plt.xlabel("Word")
plt.xlim(0,10)
plt.show()










# Compare length of quote to isTrue (Create Bins of Some sort)







# TODO Also Determine if Person is male or female.
# TODO Setiment Analysis of posts.
# TODO Shuffle Data. 











