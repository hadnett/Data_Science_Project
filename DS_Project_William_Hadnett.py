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
for i in range(1,2,1):
    url= 'https://www.politifact.com/factchecks/list/?page='+str(i)+'&ruling=true'
    html = requests.get(url).text
    soup = BeautifulSoup(html,	'html5lib')
    soup = soup.find_all('li',class_ = 'o-listicle__item')
    for post in soup:
        quote = post.find("div", class_ = 'm-statement__quote').text.strip()
        author = post.find("a", class_ = 'm-statement__name').text.strip()
        post_details.append([author, quote])