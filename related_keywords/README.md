Related Keyword Extractor
=========================

given .ipynb file is a source code for extracting topN most related keywords for given keywords from input texts.

What you need 
---------
konlpy ofr morphorlogy analysis is needed to extract nouns 

> Reference http://konlpy-ko.readthedocs.io/ko/v0.4.3/

pip install konlpy

Need input text data
-------------
For this hackathon, we used Twitter data crawled from 2019.03.05 to 2019.04.17 

*Data will not be uploaded, but you can crawl the data yourself with twitter REST API

Usage
--------------
Given time variable is set from the clicked time by an user.
Tweets published within 7 days will be considered.
If the keyword is put, the most frequent nouns that come up together with the given keywords are returned in descending order.


Lisense
--------------
Team. 퇴근후강남역



