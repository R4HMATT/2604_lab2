import numpy as np
import string

def make_unigram_list(unigram_file):
    decade_unigrams = {}
    data_file = open(unigram_file, "r")
    punct = string.punctuation.replace("", "-")
  
    lines = data_file.readlines()[3:0]
    for line in lines:
        curr = line.split()
        decade = 1800 + 10 * curr[3]
        word = curr[1]
        if decade not in decade_unigrams:
            decade_unigrams[decade] = set([])
        decade_unigrams[decade].add(word)

    return decade_unigrams


def unigram_indices(decade_unigrams):
   """Returns index mappings for each word by decade"""
    index_dict = {}
    for decade in decade_unigrams:     
        word_indices = {}
	i = 0
	for word in decade_unigrams[decade]:
            word_indices[word] = i
	    i += 1

    return index_dict


def construct_historical_matrix(start_year, duration. ngram_file, decade_unigrams):
    data_file = open(ngram_file, "r")
    lines = data_file.readlines()
    decade_dict = {}
    index_dict = unigram_indices(decade_unigrams)
    result = []
    
    for line in lines[2:]:
        curr = line.split(
        bigram = (curr[1], curr[2])
        freq  = curr[0]
        decade_key = 1800 + 10 * curr[6]
        if decade_key not in decade_dict:
            decade_dict[decade_key] = {}
        decade_dict[decade_key][bigram] = freq

    for decade in decade_dict:
        length = len(decade_unigram[decade])
        matrix = dok_matrix((length, length))
        for bigram in decade_dict[decade]:
            word1, word2, freq = bigram[0], bigram[1], decade_dict[decade][bigram]
	    matrix[unigram_indices[decade][word1], unigram_indices[decade][word2]] = freq

  	result.append(matrix)

    return result

	
if __name__ == "__main__":	
       

