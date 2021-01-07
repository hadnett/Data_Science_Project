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
from wordcloud import WordCloud
import string


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
freshData = False

def parseHTML(isTrue):
    for i in range(1,40,1):
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

if freshData:
    # Get True News Stories
    parseHTML('true')
    
    # Get Fake News Stories
    parseHTML('false')
    
    # 2310 posts retrieved.
    
    df = pd.DataFrame(post_details, columns=['Source', 'Quote', 'Date', 'isTrue'])
    
    # Write data to CSV file for future use. 
    df.to_csv('fake_news.csv', encoding="utf-8", index=False)

df = pd.read_csv('fake_news.csv')

# =============================================================================
# STEP 2 - Data Mining - Specify data characteristics. 
# =============================================================================

df.info()

#  #   Column  Non-Null Count  Dtype 
# ---  ------  --------------  ----- 
# 0   Source  2310 non-null   object
# 1   Quote   2310 non-null   object
# 2   Date    2310 non-null   object
# 3   isTrue  2310 non-null   bool  
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
# count             2310  ...   2310
# unique             830  ...      2
# top     Facebook posts  ...  False
# freq               490  ...   1170

# While some of the data is omitted in the terminal if we view the variable
# 'describe' in the Variable Explorer window we can see that there are 2310
# records in total. There are 830 unique soruces, 2308 unique quotes, 2013 
# unique dates and 2 unique response variables and this all appears to be 
# reasonable. The top source was 'Facebook Posts' appearing 490 times, the
# top quote appeared 2 times and the top date was the 4th November 2020. 
# Note 'top' is similiar to the mode as it represents the data which appears
# most often in a column.

# =============================================================================
# STEP 2 - Data Mining - Identify Variables  
# =============================================================================

# isTrue - Response Variable - Categorical - Classification Model Required
# Source  - Categorical
# Date - Can be both numerical and categorical
# Quote - Predictor Variable - Text - Handled as numerical for modelling.

# While there are currently only 3 predictor variables I intend to introduce
# more variables to help with this analysis during the Feature Engineering 
# stage of this project. I believe that variables such as gender, sentiment,
# length of text and most frequent words used for both fake and real news 
# may help to solve the problems outlined during this project.

# =============================================================================
# STEP 3 - Data Cleaning
# =============================================================================

# Remove quotation marks from quotes in dataframe.

# Discovered that two different types of quotes where present within this dataset
# removed both sets to avoid issues with processing later during this project.
df['Quote'] = df['Quote'].str.strip('[",“,”]') 
df['Quote'] = df['Quote'].astype(str).str.replace('[",“,”]', '')

# Convert all quotes & sources to lower case for future analysis and equality 
# checking. 

df['Quote'] = df['Quote'].apply(lambda x: x.lower())
df['Source'] = df['Source'].apply(lambda x: x.lower())

# Remove all punctuation as it adds no value to the analysis.

def remove_puncuation(data):
    text_list = [char for char in data if char not in string.punctuation]
    clean_text = ''.join(text_list)
    return clean_text

df['Quote'] = df['Quote'].apply(remove_puncuation)
df['Source'] = df['Source'].apply(remove_puncuation)

# Remove stop words from the dataset. Stop words are the most common words that
# appear in the English language (the, be, to). These words add no value to my 
# analysis as this model should only focus on key words which define meaning 
# within the text.

stop = stopwords.words('english') 
newStopWords = ['says', 'say', 'one', 'people', 'said', 'since', 'new', 'shows', 'would']
stop.extend(newStopWords)

df['Quote'] = df['Quote'].apply(lambda x: 
           ' '.join([word for word in str(x).split() if word not in (stop)])) 


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

# Check that only True and False exist in the isTrue column.
print(df.isTrue.value_counts())

# =============================================================================
# STEP 4 - Data Exploration - Univariate
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
numberTrueFalse.plot.pie(autopct='%1.2f')
plt.show()
# Split: False = 0.5065, True = 0.4935
# Balanced data is important for generating a classification model according to:
# https://www.r-bloggers.com/2020/06/why-balancing-your-data-set-is-important/#:~:text=From%20the%20above%20examples%2C%20we,set%20for%20a%20classification%20model.
# It is clear that the data selected for this project has a good balance of 
# reliable and unreliable news quotes. 

# Find max and min date: 
dateMin = df.Date.min()
# dateMin = 2012-05-22
dateMax = df.Date.max()
# dateMax = 2020-12-27

# Get Average Length of Quote
averageLength = df['Quote'].apply(len).mean()
print(averageLength)
# averageLength = 73.00389948006932

# Shortest Quote: 
shortestQuote = df['Quote'].apply(len).min()
print(shortestQuote)
# shortestQuote = 13

# Longest Quote:
longestQuote = df['Quote'].apply(len).max()
print(longestQuote)
# longestQuote = 251 

# Top 15 most commons words used in quotes:

frequentWords = df['Quote'].str.split(expand=True).stack().value_counts()

frequentWords.plot.bar()
plt.title("Top 15 Words")
plt.ylabel("Total Appearances")
plt.xlabel("Word")
plt.xlim(0,15)
plt.show()

# We can see from the bar chart displayed above that the majority of the most 
# frequently used words relate to US politicians. Trump appears 186 times 
# throughout the dataset with Donald appearing 116 times. Surprisingly
# COVID-19 only appeared 93 times. Currently this analysis of the most 
# frequently used words does not offer much insight into the business problem.
# However, in the next section I intend to review which of these words are 
# assosicated with reliable and unreliable news quotes. 

# Discovered how to create a wordcloud to visually display the top 100 words in
# the dataset via geeksforgeeks.org:
# https://www.geeksforgeeks.org/generating-word-cloud-python/
# The words increase and decrease in size depending on the number of times 
# they appear within the dataset.

all_words = ' '.join([text for text in df['Quote'] if text not in (stop)])
wordcloud = WordCloud(max_words=100, width= 800, height= 500, max_font_size = 110,
 collocations = False).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# =============================================================================
# STEP 4 - Data Exploration - Bivariate
# =============================================================================

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

statsReliability.plot.pie(autopct='%1.2f')
plt.title('Quotes Containing Digits')
plt.show()

# It appears that news quotes that contain digits (stats) are more likely to
# to be True. This is surprising. I thought that unreliable quotes
# would have used stats to leverage peoples opinions. 

# Top 10 Frequently used words in reliable and unreliable news quotes.

reliableData = df[df["isTrue"] == True]

frequentWordsReliable = reliableData.Quote.str.split(expand=True).stack().value_counts()
        
frequentWordsReliable.plot.bar()
plt.title("10 Most Frequent Words in Reliable News")
plt.ylabel("Count")
plt.xlabel("Word")
plt.xlim(0,10)
plt.show()

# The most frequently used word for reliable new quotes is percent and it appears
# 143 times throughout the dataset. Other words associated with reliable news quotes
# are Trump, tax and states.
     
unreliableData = df[df["isTrue"] == False]

frequentWordsUnreliable = unreliableData.Quote.str.split(expand=True).stack().value_counts()

frequentWordsUnreliable.plot.bar()
plt.title("10 Most Frequent Words in Unreliable News")
plt.ylabel("Count")
plt.xlabel("Word")
plt.xlim(0,10)
plt.show()

# The most frequently used word for unreliable new quotes is trump and it appears
# 139 times throughout the dataset. Other words associated with unreliable news quotes
# are biden, joe, donald and coid19. It appears as if unreliable quotes tend to target
# individuals i.e. Donald Trump and Joe Biden more so than reliable quotes. This 
# could suggest that unreliable news is trying to damage an individuals standing.

mean_len = df.Quote.str.len().groupby(df.isTrue).mean()

mean_len.plot.bar()
plt.title("Average Lenght of Quote by Reliability")
plt.ylabel("Average Length")
plt.xlabel("Reliability")
plt.show()

# The average length of reliable quotes is 75.262511 and the average length
# of unreliable quotes is 70.803251. This would suggest that on average the 
# more reliable news quotes are typically longer in length.

# =============================================================================
# STEP 4 - Data Exploration - Multivariate
# =============================================================================

# Currently I believe that multivariate Analysis cannot be carried out at this
# stage due to a lack of variables (All Categorical). However, in the next section 'Feature 
# Engineering' I intend to create more varaibles to further my analysis and 
# it will be in this section that I continue exploring the data and conducting
# multivariate analysis.

# =============================================================================
# STEP 5 - Feature Engineering
# =============================================================================

# TextBlob used to derive sentiment from the quotes gathered:
# https://textblob.readthedocs.io/en/dev/
from textblob import TextBlob

df['Sentiment'] = df.Quote.apply(lambda x: TextBlob(str(x)).sentiment.polarity)
# 0 = Netural Setiment, Greater than 0 indicates position, Less than 0  indicates
# negative.

positive = df.groupby('isTrue')['Sentiment'].apply(lambda x: x[x > 0].count())
neutral = df.groupby('isTrue')['Sentiment'].apply(lambda x: x[x == 0].count())
negative = df.groupby('isTrue')['Sentiment'].apply(lambda x: x[x < 0].count())

plotDf = pd.merge(pd.merge(positive, neutral, on='isTrue', suffixes=[None, '_neutral']), negative, on='isTrue', suffixes=['_positive', '_negative'])
plotDf.plot.bar()
plt.title("Quotes by Sentiment")
plt.ylabel("Count")
plt.xlabel("Relability")
plt.show()

# This plot is surprising, originally I expected that unreliable news would be more
# negative to provoke negative emotions and cause fear. However, the data tells us
# that 702 unreliable quotes are actually deemed as neutral, 243 positive and 224
# negative. While reliable news has 237 negative quotes, 326 positive quotes and 576 
# netural quotes. This would suggest that fake news is not being spread for the purposes
# of fear or to provoke a negative emotion response. It does not however, rule out 
# other causes such as pushing corrupted political agendas or damaging a person/businesses
# reputation. 

# Get Gender of Source.
import openapi_client 

# Check to determine whether or not a source (name) is valid. This function 
# was written to prevent sources such as Facebook post and Tweet from being assigned a 
# gender as we do not know the name of the poster.
def checkValidName(name):
    # The en_core_web_sm should install with the spacy library. However, in my case
    # it did not. If you experience issues running this import please open your enviroment
    # terminal in the anaconda navigator and run conda install -c conda-forge spacy-model-en_core_web_sm.
    import en_core_web_sm
    nlp = en_core_web_sm.load()
    doc = nlp(name)
    for token in doc.ents:
        if token.label_ == 'PERSON':
            return True
    return False

# Please note on large datasets this function can take several minutes to run.
def getGender(name):
    # The code within this function was taken from the namesor SDK located on github:
    # https://github.com/namsor/namsor-python-sdk2/blob/master/docs/PersonalApi.md#gender_full
    # However, the code has been adapted slightly to manage unknown genders and to return 
    # only the gender field and not the entire json object.
    from openapi_client.rest import ApiException
    
    # This key still has approximately 4170 credits left.
    configuration = openapi_client.Configuration()
    configuration.api_key['X-API-KEY'] = '57ce2a2f19dd0ec20baca43afddaef72'
    
    
    api_instance = openapi_client.PersonalApi(openapi_client.ApiClient(configuration))
    full_name = name
    
    try:
        if checkValidName(full_name):
            api_response = api_instance.gender_full(full_name)
            return api_response.likely_gender
        else:
            return 'unknown'
    except ApiException as e:
        print("Exception when calling PersonalApi->gender: %s\n" % e)

if freshData:
    df['Gender'] = df.Source.apply(lambda x: getGender(str(x)))
    # Output new features to file for future use as retrieving gender from 
    # API can take up to several minutes to complete and as I am using a free
    # account I am only assigned 5000 credits a month.
    df.to_csv('fake_news_features.csv', encoding="utf-8", index=False)

df = pd.read_csv('fake_news_features.csv')
    

genderCount = df['Gender'].value_counts()
genderCount.plot.pie(autopct='%1.2f')
plt.show()

# 45.97% of the quotes collected the gender is not known for and these account
# primarily for quotes where the source is documented as Facebook Post, Blogger
# Intsagram i.e. posts of social media platforms/websites. 43.15% for the quotes collected 
# belong to males and only 10.88% of the quotes collected belong to females.

genderTrue = df.groupby('Gender')['isTrue'].apply(lambda x: x[x == True].count())

genderTrue.plot.bar()
plt.title("Reliable Quotes by Gender")
plt.ylabel("Count")
plt.xlabel("Gender")
plt.show()

genderFalse = df.groupby('Gender')['isTrue'].apply(lambda x: x[x == False].count())

genderFalse.plot.bar()
plt.title("Unreliable Quotes by Gender")
plt.ylabel("Count")
plt.xlabel("Gender")
plt.show()

# Upon reviewing the graphs above it is clear that males are the most reliable
# gender category and the unknown category is the least reliable. However, this 
# is not a fair statement due to the unbalanced nature of the gender categories.
# In order to make this a fair statement more data would have to be collected 
# from female sources to compare reliability with gender. 

#### Calculate Word Count for each quote ####

df['Word_Count'] = df.Quote.apply(lambda x: len(str(x).split(' ')))

meanWordCount =  df.Word_Count.mean()
print(meanWordCount)
# meanWordCount: 17.620883882149048

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(x=df.Word_Count).set_title("Word Count")
plt.show()

# Upon reviewing this boxplot of the Word_Count column it is clear that
# there are outliers present. However, I do not believe that these outliers
# are so significant that they will basis the model produced. Therefore, 
# for the purposes of this project the outliers will be kept. 

figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(df.corr(), annot=True, cmap = 'Reds')
plt.show()

# After reviewing the heatmap above there appears to be a strong negative correlation 
# with the year variable (-0.77). I believe this is caused by the unbalanced nature 
# of the date. The majority of the quotes collected are from the year 2020. The next
# strongest correlation appears to be the month of the quote with a negative 
# correlation of (-0.15). Sentiment produces a positive correlation of 0.11 and 
# the Word_Count produced above has correlation with the response variable of 0.072.
# There appears to be no multicollinearity within the data.

# Encode isTrue column for processing:
df['isTrueType']=np.where(df.isTrue == True,1,0)

# =============================================================================
# STEP 4 - Data Exploration - Multivariate
# =============================================================================

result = pd.pivot_table(data=df, index='isTrue', columns='Gender', values='Word_Count', aggfunc=np.mean)
print(result)

# Gender     female       male    unknown
# isTrue                                 
# False   15.974359  17.571970  16.933025
# True    18.688679  18.381148  17.056410

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(result, annot=True, cmap = 'RdYlGn', center=0.117)
plt.show()

# The heatmap produced shows that females with reliable data are more likely to 
# produce longer quotes. However, females with unreliable data are more likely to produce
# shorter quotes.

result = pd.pivot_table(data=df, index='isTrue', columns='Gender', values='Sentiment', aggfunc=np.mean)
print(result)

# Gender    female      male   unknown
# isTrue                              
# False   0.061289  0.028853  0.016749
# True    0.078845  0.074304  0.029842

figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(result, annot=True, cmap = 'RdYlGn', center=0.117)
plt.show()

# This heatmap shows that unknown sources who are unreliable are 
# more likely to have a lower average sentiment. This could mean 
# that they are related to more negative news quotes. Female sources
# who are reliable have the highest sentiment value and this could infer
# that these individuals are the source of more positive news quotes.

result = pd.pivot_table(data=df, index='Gender', columns='Year', 
                        values='isTrue', aggfunc='count', fill_value=0)
print(result)

# Year     2012  2013  2014  2015  2016  2017  2018  2019  2020
# Gender                                                       
# female      0    30    23    28    55    24    17    25    49
# male        1    91   125   114   128    71    77   125   264
# unknown     0    29    31    43    30    13    29   140   746

figure(num=None, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(result, annot=True, cmap = 'RdYlGn', center=0.117)
plt.show()

# This heatmap shows an interesting trend that I identified in the data.
# In the years 2016 and 2020 news quotes from all three genders increased.
# One possible reason for this is the US Presedential Elections.   

result = pd.pivot_table(data=df, index='isTrue', columns='Year', 
                        values='isTrueType', aggfunc='count', fill_value=0)
print(result)

# Year    2012  2013  2014  2015  2016  2017  2018  2019  2020
# isTrue                                                      
# False      0     1     0     2     1     4    13   188   960
# True       1   149   179   183   212   104   110   102    99

figure(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(result, annot=True, cmap = 'RdYlGn', center=0.117)
plt.show()

# Interestingly when the data is grouped by isTrue and Year we can see that in 
# 2016 the number of reliable new quotes increased but in 2020 then number of 
# of unreliable news quotes increased and the reliable news quotes actually 
# decreased.


# =============================================================================
# STEP 6 - Predictive Modelling - Split Data
# =============================================================================

# The business problem I want to solve is whether or not fake news can be detected
# via the contents of the news quote. Therefore, the model will be provided with 
# the news quote and whether or not the new quote is true or false.
x = df['Quote']
y = df['isTrueType']

from sklearn.model_selection import train_test_split

# Data split into three seperate groups so that models can be trained and validated.
# Once validated the best model will be selected to run against the test data.

train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

# =============================================================================
# STEP 6 - Predictive Modelling - Prediction Models (Fit and Validate)
# =============================================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

# TF-IDF = Term Frequency (TF) * Inverse Document Frequency (IDF)
# Discovered TF-IDF while researching how to use textual data in a predictive 
# model: https://stackabuse.com/text-classification-with-python-and-scikit-learn/

# Sklearn models cannot accept text data as features. Therefore, the text in each
# quote has to be converted to a number. In order to do this I used TFIDFVectorizer
# this alogrithm assigns a weight to each word within a document/corpus and this 
# weight represents this words importance. This weighting can then be passed 
# to the model as a feature. Another alternative that I encountered was CountVectorizer
# this function simply counts the number of times a word appears within the document. 
# However, I decided against using this function as basis can be given to the most 
# frequently used words. 
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_val = tfidf_vectorizer.transform(x_val)
tfidf_test = tfidf_vectorizer.transform(x_test)

########## Model 1: Passive Aggressive Classifier ###########

# The PassiveAggressiveClassifier is a model that is commonly used for spam filtering
# and even fake news detection as it is particularly good at handling large datasets. 
# For example, retrieving large quanities of news from Twitter. It learns by being 
# passive when it makes a correct prediction, but aggressive when it makes a wrong 
# prediction (change made to the model). This model may not be perfectly suited to 
# the size of dataset that is being used in this project. However, as it is one of the
# most commonly used models for fake news detection I was keen to see how it would perform.
# https://www.geeksforgeeks.org/passive-aggressive-classifiers/

from sklearn.linear_model import PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

###  Evaluate Model Based on Validation Set ###

predictions = pac.predict(tfidf_val)

confusionMatrix = confusion_matrix(y_val, predictions)
print(confusionMatrix)

accuracy = (confusionMatrix[0,0]+confusionMatrix[1,1])/len(predictions)
errorRate = 1- accuracy
precision = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[0,1])
recall = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[1,0])
print("Accuracy: " + str(accuracy))
print("Error Rate: " + str(errorRate))
print("Precision: " + str(precision))
print("Recall: " + str(recall))

# [[129  44]
# [ 49 124]]

# Accuracy: 0.7312138728323699
# Error Rate: 0.26878612716763006
# Precision: 0.7380952380952381
# Recall: 0.7167630057803468

########## Model 2: MLPClassifier ###########

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

model = MLPClassifier()

#GridSearchCV is imported from sklearn and it is used to run a brute force search to 
#determine the best parameters to use for model in this case my neural network. The 
#check_parameters json is populated with possible parameters and these parameters and
#run against the training data using the fit method. fit() returns the best parameters
#for the model in a key value pair. 
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
parameter_space = {
    'hidden_layer_sizes': [(10,30,10),(20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
gridsearchcv = GridSearchCV(model, parameter_space, n_jobs=-1, cv=5)
gridsearchcv.fit(tfidf_train, y_train)

model = MLPClassifier(hidden_layer_sizes=gridsearchcv.best_params_['hidden_layer_sizes'], activation=gridsearchcv.best_params_['activation'], alpha=gridsearchcv.best_params_['alpha'], 
                      learning_rate=gridsearchcv.best_params_['learning_rate'], solver=gridsearchcv.best_params_['solver'], 
                      max_iter=500)

model.fit(tfidf_train, y_train)

###  Evaluate Model Based on Validation Set ###

predictions = model.predict(tfidf_val)

confusionMatrix = confusion_matrix(y_val, predictions)
print(confusionMatrix)

accuracy = (confusionMatrix[0,0]+confusionMatrix[1,1])/len(predictions)
errorRate = 1- accuracy
precision = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[0,1])
recall = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[1,0])
print("Accuracy: " + str(accuracy))
print("Error Rate: " + str(errorRate))
print("Precision: " + str(precision))
print("Recall: " + str(recall))

# [[133  40]
# [ 44 129]]

# Accuracy: 0.7572254335260116
# Error Rate: 0.24277456647398843
# Precision: 0.7633136094674556
# Recall: 0.7456647398843931

########## Model 3: LogisticRegression ###########

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(tfidf_train, y_train)

###  Evaluate Model Based on Validation Set ###

predictions = model.predict(tfidf_val)

confusionMatrix = confusion_matrix(y_val, predictions)
print(confusionMatrix)

accuracy = (confusionMatrix[0,0]+confusionMatrix[1,1])/len(predictions)
errorRate = 1- accuracy
precision = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[0,1])
recall = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[1,0])
print("Accuracy: " + str(accuracy))
print("Error Rate: " + str(errorRate))
print("Precision: " + str(precision))
print("Recall: " + str(recall))

# [[140  33]
# [ 49 124]]

# Accuracy: 0.7630057803468208
# Error Rate: 0.23699421965317924
# Precision: 0.7898089171974523
# Recall: 0.7167630057803468

########## Model 4: MultinomialNB ###########

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(tfidf_train, y_train)
MultinomialNB()

###  Evaluate Model Based on Validation Set ###

predictions = clf.predict(tfidf_val)

confusionMatrix = confusion_matrix(y_val, predictions)
print(confusionMatrix)

accuracy = (confusionMatrix[0,0]+confusionMatrix[1,1])/len(predictions)
errorRate = 1- accuracy
precision = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[0,1])
recall = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[1,0])
print("Accuracy: " + str(accuracy))
print("Error Rate: " + str(errorRate))
print("Precision: " + str(precision))
print("Recall: " + str(recall))

# [[148  37]
# [ 19 142]]

# Accuracy: 0.838150289017341
# Error Rate: 0.161849710982659
# Precision: 0.7932960893854749
# Recall: 0.8819875776397516

# =============================================================================
# STEP 6 - Predictive Modelling - Model Selection
# =============================================================================

# We can see that from the analysis above that all four models performed similarly.
# However, it is clear that model 4 (MultinomialNB) performed the best out of 
# the four models tested. Model 4 has a accuracy of 76.87%, a precision of 77.51% 
# and a recall  of 75.72%. These figures give me confidence in this model and I 
# believe that it has the potential to reliably detect whether or not a news quote
# is reliable or unreliable (True or False).

predictions = clf.predict(tfidf_test)

confusionMatrix = confusion_matrix(y_test, predictions)
print(confusionMatrix)

labels = ['Reliable', 'Unreliable']
plot_confusion_matrix(clf,tfidf_test, y_test, display_labels=labels, cmap=plt.cm.Blues) 

accuracy = (confusionMatrix[0,0]+confusionMatrix[1,1])/len(predictions)
errorRate = 1- accuracy
precision = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[0,1])
recall = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[1,0])
F1Score = 2 * ((precision * recall) / (precision + recall))
print("Accuracy: " + str(accuracy))
print("Error Rate: " + str(errorRate))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1 Score: " + str(F1Score))

# [[95 23]
# [23 90]]

# Accuracy: 0.8008658008658008
# Error Rate: 0.19913419913419916
# Precision: 0.7964601769911505
# Recall: 0.7964601769911505
# F1 Score: 0.7964601769911505

### Plot ROC Curve ###  
fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Fake News Detection Curve')
plt.legend()
plt.show()

# Receiver Operating Characteristics Curve.
# This curve tends towards the top left hand corner of the plot and this is common
# in better performing classification models. The blueline in the middle of the plot
# denotes a model that is no more accurate than random choice.
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

'''
I believe that the model produced performs well and that it can predict whether 
or not news is reliable or unreliable approximately 80% for the time. The model used 
is a Multinomial Naive Bayes classification model. This type of classification model 
has one major limitation and that is that all attributes are treated equally and therefore
contribute equally to the predictions made by the model. Therefore, it is possible that
stronger attributes, in this case word weightings could be bringing down the 
overall performance score. However, after comparing this model with three other models 
I believe it is the best model to use based on the data at hand. 

Whether or not this model is used then becomes a business decision. While I believe 
this model should not have the final say as to whether or not news is reliable I do believe
it could be used on digital platforms such as Facebook to tag posts with 'The news
you are reading may be unreliable' so that the user themselves can then continue 
to fact check the article or post. 

In conclusion, I believe I have answered the busniess problem outlined at the start
of this project. It is possible to detect whether or not news is fake based of the
contents of the news quote/text. 
'''






