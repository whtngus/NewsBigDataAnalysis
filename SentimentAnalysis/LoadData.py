# -*- coding: utf-8 -*-
"""
@ source: https://cyc1am3n.github.io/2018/11/10/classifying_korean_movie_review.html
@ modified by Hyung-Kwon Ko
@ since: Tue Apr 30 19:46:23 2019
"""

import os

class LoadData:
    def __init__(self,address):
        '''
        @param address: directory where data file exists
        @param filename: data file name
        '''
        self.address = address

    def read_data(self,filename):
        '''Function to load data
        '''
        os.chdir(self.address)
        with open(filename, 'r', encoding='UTF8') as f: # set UTF8 since it's Korean
            data = [line.split('\t') for line in f.read().splitlines()]
            data = data[1:] # remove header
        return data
