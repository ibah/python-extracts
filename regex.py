# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 12:22:40 2016

@author: msiwek
"""

# 2017
x = 'Braund, Mr. Owen Harris'
x.split(',')
x.replace(',','|')
x.replace('[A-Z]','|') # no regex
import re
re.findall('.*,', x)
re.search('.*,', x).group()
re.findall('[A-Za-z]*,', x)
re.findall('[A-Za-z]+', x)
re.findall('\. O',x)
re.sub('.*, ','',x)
re.sub('\..*','',x)
re.search('(.*, )|(\..*)',x).group()
re.search('(.*, )|(\..*)',x) # match object
re.sub('(.*, )|(\..*)','',x) # getting just the title

# fixing pandas column names
import numpy as np
import pandas as pd
df = pd.DataFrame(np.random.randn(5,3), columns=['a[b','c<d','e]f'])
features = pd.Series(df.columns)
features = features.apply(lambda x: re.sub('[\[\]<]', '-', x))
df.columns = features
df

# parsing URLs

# URL parameters

import urllib.parse as urlparse
# url = 'http://example.com/?q=abc&p=123'
parsed = urlparse.urlparse(url)
params = urlparse.parse_qsl(parsed.query)
for x,y in params:
    print("Parameter = "+x,"Value = "+y)


# 2016

import re
str = 'an example word:cat!!'
match = re.search(r'word:\w\w\w', str)
# If-statement after search() tests if it succeeded
if match:  # else is None
    print 'found', match.group()  # 'found word:cat'
else:
    print 'did not find'

# email address:
# this doesn't catch - and .
str = 'purple alice-b@google.com monkey dishwasher'
match = re.search(r'\w+@\w+', str)
if match:
    print match.group()  # 'b@google'
# the problem fixed:
match = re.search(r'[\w.-]+@[\w.-]+', str)
if match:
    print match.group()  # 'alice-b@google.com'
# extract the username and host separately
match = re.search('([\w.-]+)@([\w.-]+)', str)
if match:
    print match.group()   ## 'alice-b@google.com' (the whole match)
    print match.group(1)  ## 'alice-b' (the username, group 1)
    print match.group(2)  ## 'google.com' (the host, group 2)

## Suppose we have a text with many email addresses
str = 'purple alice@google.com, blah monkey bob@abc.com blah dishwasher'
## Here re.findall() returns a list of all the found email strings
emails = re.findall(r'[\w\.-]+@[\w\.-]+', str) ## ['alice@google.com', 'bob@abc.com']
for email in emails:
    # do something with each found email string
    print email

# findall() wih files
# Open file
f = open('test.txt', 'r')
# Feed the file text into findall(); it returns a list of all the found strings
strings = re.findall(r'some pattern', f.read())

# findall() with groups
str = 'purple alice@google.com, blah monkey bob@abc.com blah dishwasher'
tuples = re.findall(r'([\w\.-]+)@([\w\.-]+)', str)
print tuples  ## [('alice', 'google.com'), ('bob', 'abc.com')]
for tuple in tuples:
    print tuple[0]  ## username
    print tuple[1]  ## host

# Perl Compatible Regular Expressions -- pcre
str = r'<b>foo</b> and <i>so on</i>'
pat = r'<.*>'
# greedy (default) - wrong
match = re.match(pat, str)
if match:
    print match.group()
match = re.findall(pat, str)
if match:
    print match
# non-greedy - correct
pat = r'<.*?>'
match = re.findall(pat, str)
if match:
    print match

# subsitution
str = 'purple alice@google.com, blah monkey bob@abc.com blah dishwasher'
## re.sub(pat, replacement, str) -- returns new string with all replacements,
## \1 is group(1), \2 group(2) in the replacement
print re.sub(r'([\w\.-]+)@([\w\.-]+)', r'\1@yo-yo-dyne.com', str)
## purple alice@yo-yo-dyne.com, blah monkey bob@yo-yo-dyne.com blah dishwasher

4.9*1.4-7/1.4








