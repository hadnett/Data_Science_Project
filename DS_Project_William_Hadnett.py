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


# TODO Data Cleaning Remove Data String. 
# TODO Also Determine if Person is male or female.
# TODO Setiment Analysis of posts.
# TODO Export Results to CSV file.
# TODO Shuffle Data. 











