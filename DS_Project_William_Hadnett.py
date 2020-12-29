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
# STEP 1 - Data Mining - Specify data characteristics. 
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
# STEP 1 - Data Mining - Identify Variables  
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
# STEP 2 - Data Cleaning
# =============================================================================

# Remove quotation marks from quotes in dataframe.

# Discovered that two different types of quotes where present within this dataset
# removed both sets to avoid issues with processing later during this project.

df['Quote'] = df['Quote'].str.strip('[",“,”]') 
df['Quote'] = df['Quote'].str.replace('[",“,”]', '')












# TODO Data Cleaning Remove Data String. 
# TODO Also Determine if Person is male or female.
# TODO Setiment Analysis of posts.
# TODO Export Results to CSV file.
# TODO Shuffle Data. 











