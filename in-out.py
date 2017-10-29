# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:36:05 2017
Input-Ouput operations in Python
(extracts)
"""

"""

File system

"""


import os
import numpy as np
import pandas as pd
import datetime
# interactive work wd:
os.getcwd()
# runtime wd:
print(os.path.dirname(os.path.realpath(__file__))) # works only during a runtime

os.path.join('../input', 'test.csv')
os.path.join('..', 'input', 'test.csv') # better
os.path.join('test.csv')

# serializing objects into files
x = list('Hello World!')
import pickle
os.getcwd()
with open('example.pkl','wb') as file:
    pickle.dump(x,file)
with open('example.pkl','rb') as file:
    x_new = pickle.load(file)
print(x_new)


# looping over files

# no recursion
path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '../input/')
for file_name in os.listdir(path):
    print(file_name)
# with recursion -> see on stackoverflow


# gzip files

# create
import gzip
content = "Lots of content here"
f = gzip.open('Onlyfinnaly.log.gz', 'wb')
f.write(content)
f.close()
# read from
import gzip
f=gzip.open('Onlyfinnaly.log.gz','rb')
file_content=f.read()
print(file_content)


# TXT
# <----------------------------------- fix this please
now = datetime.datetime.now()
then = now + datetime.timedelta(days=9)
df = pd.DataFrame(np.random.random((10,5)),
                  pd.date_range(now.strftime('%Y-%m-%d'), then.strftime('%Y-%m-%d')),
                  list('abcde'))
os.getcwd()
with open('example_text.txt','w') as file:
    np.savetxt(file,df.values, '%f')
    
x = y = z = np.arange(0.0,5.0,1.0)
np.savetxt('test.txt', x, delimiter=',')


"""
Further explanation of the `fmt` parameter
(``%[flag]width[.precision]specifier``):

flags:
    ``-`` : left justify

    ``+`` : Forces to precede result with + or -.

    ``0`` : Left pad the number with zeros instead of space (see width).

width:
    Minimum number of characters to be printed. The value is not truncated
    if it has more characters.

precision:
    - For integer specifiers (eg. ``d,i,o,x``), the minimum number of
      digits.
    - For ``e, E`` and ``f`` specifiers, the number of digits to print
      after the decimal point.
    - For ``g`` and ``G``, the maximum number of significant digits.
    - For ``s``, the maximum number of characters.

specifiers:
    ``c`` : character

    ``d`` or ``i`` : signed decimal integer

    ``e`` or ``E`` : scientific notation with ``e`` or ``E``.

    ``f`` : decimal floating point

    ``g,G`` : use the shorter of ``e,E`` or ``f``

    ``o`` : signed octal

    ``s`` : string of characters

    ``u`` : unsigned decimal integer

    ``x,X`` : unsigned hexadecimal integer
"""



# CSV

import csv as csv

# General remarks
csv.list_dialects()
# csv.reader(csvfile, dilect) # works for any iterator

# Reading

# Open-Close
csv_file = open('csv/train.csv') # 'r' is default
csv_file_object = csv.reader(csv_file)
header = csv_file_object.__next__()
data = []
for row in csv_file_object:
    data.append(row)
csv_file.close()

# with(open)
with open('csv/train.csv') as train_file:
    train_reader = csv.reader(train_file)
    header = train_reader.__next__()
    for row in train_reader:
        data.append(row)


# Writing

# Open-close
csv_file = open("csv/genderbasedmodel.csv", "w", newline='')
# newline is needed to be platform independent (e.g. in Windows empty rows appear without it):
# If newline='' is not specified, newlines embedded inside quoted fields will not be interpreted correctly,
# and on platforms that use \r\n linendings on write an extra \r will be added.
# It should always be safe to specify newline='', since the csv module does its own (universal) newline handling.
csv_file_object = csv.writer(csv_file)
csv_file_object.writerow(["PassengerId", "Survived"])
csv_file.close()

# with
# ditto

"""
Excel

special module available
below tools form Pandas
"""
# read
df0 = pd.read_excel('input.xlsx', 'Sheet1')
# write
writer = pd.ExcelWriter('output.xlsx')
df1.to_excel(writer,'Sheet1')
df2.to_excel(writer,'Sheet2')
writer.save()




"""

Printing

"""


"""
PrettyPrinter
indent=1, width=80, depth=None, stream=None, *, compact=False
"""
from pprint import PrettyPrinter
stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
stuff
stuff.insert(0, stuff[:])
stuff
pp = PrettyPrinter(indent=4)
pp.pprint(stuff)
pp = PrettyPrinter(width=41, compact=True)
pp.pprint(stuff)
tup = ('spam', ('eggs', ('lumberjack', ('knights', ('ni', ('dead',('parrot', ('fresh fruit',))))))))
pp = PrettyPrinter(depth=6)
pp.pprint(tup)
"""
pformat
object, indent=1, width=80, depth=None, *, compact=False
pprint
object, stream=None, indent=1, width=80, depth=None, *, compact=False
"""
from pprint import pprint
from pprint import pformat
stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
stuff.insert(0, stuff)
stuff
pformat(stuff)
pprint(stuff)




"""

Logging

"""

"""
https://docs.python.org/3/howto/logging.html#logging-basic-tutorial
https://docs.python.org/3.6/howto/logging-cookbook.html
"""
# into console
import logging
logging.warning('Watch out!')  # will print a message to the console
logging.info('I told you so')  # will not print anything
# into a file
# (start new interpreter)
import os; os.getcwd(); os.chdir('G:\\Dropbox\\cooperation\_python\\Extracts')
import logging
logging.basicConfig(filename='data\\example.log',level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
# setting up the logging level in a command line
#--log=INFO
# Display date & time
import logging
logging.basicConfig(format='%(asctime)s %(message)s')
logging.warning('is when this event was logged.')
# (start new interpreter)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logging.warning('is when this event was logged.')
#
print(logging.Logger.manager.loggerDict.keys())


"""

Dates

"""


import datetime

now = datetime.datetime.now()

print
print "Current date and time using str method of datetime object:"
print str(now)

print
print "Current date and time using instance attributes:"
print "Current year: %d" % now.year
print "Current month: %d" % now.month
print "Current day: %d" % now.day
print "Current hour: %d" % now.hour
print "Current minute: %d" % now.minute
print "Current second: %d" % now.second
print "Current microsecond: %d" % now.microsecond

print
print "Current date and time using strftime:"
print now.strftime("%Y-%m-%d %H:%M")

print
print "Current date and time using isoformat:"
print now.isoformat()

"""
source: python strptime

Directive	Meaning	Notes
%a	Locale's abbreviated weekday name.	
%A	Locale's full weekday name.	
%b	Locale's abbreviated month name.	
%B	Locale's full month name.	
%c	Locale's appropriate date and time representation.	
%d	Day of the month as a decimal number [01,31].	
%H	Hour (24-hour clock) as a decimal number [00,23].	
%I	Hour (12-hour clock) as a decimal number [01,12].	
%j	Day of the year as a decimal number [001,366].	
%m	Month as a decimal number [01,12].	
%M	Minute as a decimal number [00,59].	
%p	Locale's equivalent of either AM or PM.	(1)
%S	Second as a decimal number [00,61].	(2)
%U	Week number of the year (Sunday as the first day of the week) as a decimal number [00,53]. All days in a new year preceding the first Sunday are considered to be in week 0.	(3)
%w	Weekday as a decimal number [0(Sunday),6].	
%W	Week number of the year (Monday as the first day of the week) as a decimal number [00,53]. All days in a new year preceding the first Monday are considered to be in week 0.	(3)
%x	Locale's appropriate date representation.	
%X	Locale's appropriate time representation.	
%y	Year without century as a decimal number [00,99].	
%Y	Year with century as a decimal number.	
%Z	Time zone name (no characters if no time zone exists).	
%%	A literal "%" character.
"""
import pandas as pd
from datetime import datetime
date = '23/May/2017:23:45:02 +0000'
pd.to_datetime(date, format='%d/%b/%Y:%H:%M:%S +0000') # does not support time zones????
datetime.strptime(date, '%d/%b/%Y:%H:%M:%S %z') # time zone


"""

email

"""
def send_mail(send_from, send_to, subject, text,
              server, user, pwd,
              files=None):
    assert isinstance(send_to, list)

    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(text))

    for f in files or []:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=basename(f)
            )
            part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
            msg.attach(part)
    
    smtp = smtplib.SMTP(server)
    smtp.ehlo()
    smtp.starttls()
    smtp.login(user, pwd)
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.close()

send_mail('@from',
      ['@to'],
      'Subject',
      'Content',
      'server:587',
      'user',
      'passwd',
      [some_file])


"""

mysql

"""
import pymysql.cursors

# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='user',
                             password='passwd',
                             db='db',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

try:
    with connection.cursor() as cursor:
        # Create a new record
        sql = "INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)"
        cursor.execute(sql, ('webmaster@python.org', 'very-secret'))

    # connection is not autocommit by default. So you must commit to save
    # your changes.
    connection.commit()

    with connection.cursor() as cursor:
        # Read a single record
        sql = "SELECT `id`, `password` FROM `users` WHERE `email`=%s"
        cursor.execute(sql, ('webmaster@python.org',))
        result = cursor.fetchone()
        print(result)
finally:
    connection.close()

"""

ORACLE SQL

"""

import cx_Oracle as cx
con = cx.connect('user/password@server')
print(con.version)
cur = con.cursor()
cur.execute('SELECT * FROM v$version')
# obtaining results
header = [x[0] for x in cur.description]
for result in cur:
    print(result)
row = cur.fetchone()
print(row)
res = cur.fetchmany(numRows=3)
print(res)
res = cur.fetchall()
print(res)
# closing
cur.close()
con.close()


"""

Error Handling

"""

# catch-all error handling
import sys
try:
  untrusted.execute()
except: # catch *all* exceptions
  e = sys.exc_info()#[0]
  print("Error: %s" % e)

