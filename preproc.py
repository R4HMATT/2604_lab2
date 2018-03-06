import numpy as np
import scipy
#import scipy.sparse
from scipy.sparse import *
from scipy import *
import string
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

def POS_appender(word,pos_tag):
    """pos_tag is a string that holds the pos that was tagged from the unigram/bigram file """
    appendee = ""
    
    if pos_tag[0] == 'd':
        appendee = 'det'
    if pos_tag[0] == 'n':
        appendee = 'n'
    elif pos_tag[0] == 'v':
        appendee = 'v'
    elif pos_tag[0] == 'j':
        appendee = 'a'
    elif pos_tag[0] == 'r':
        appendee = 'adv'
    


    #appendee = pos_tag[0]
    elif pos_tag == 'to':
       appendee = 'infinitive-marker'
    elif pos_tag[0:2] == "vm":
       appendee = 'modal'
    elif pos_tag[0] == "a":
        appendee = 'det'
    elif pos_tag[0] == 'i':
        appendee = 'prep'
    elif pos_tag[0] == 'p':
        appendee = 'pron'
    elif pos_tag[0:2] == 'uh':
        appendee = 'interjection'
    elif pos_tag[0] == 'c':
        appendee = 'conj'
    elif pos_tag[0:2] == 'ex':
        appendee = 'pron'
    #what about FO and MC
    if appendee != "":
        mod_word = word + " " + appendee
    else:
        mod_word = word

    return mod_word


def make_unigram_list(unigram_file):

	#~~~~~~~POS MODIFICATIONS
    #nltk.download('wordnet')
    wnl = WordNetLemmatizer()
    decade_unigrams = {}
    data_file = open(unigram_file, "r")
    punct = string.punctuation.replace("", "-")
  
    lines = data_file.readlines()[3:]
    data_file.close()
    assert len(lines) > 0
    for line in lines:
        curr = line.split()
        if curr[5][0] not in "0123456789":
            continue
        
        decade = 1800 + 10 * int(curr[5])
        #~~~~~~~~~~lemmatizer
        #~~~~~once we import the POS tags, we give into lemmatizer an optional
        # parameter for a specific pos to lemmatize for the word in curr[1]
        # optional parameter either (strings) a,n,r,v  where r is adverb I believe
        pos1 = curr[3]
        pos2 = curr[4]

        if pos1[0] not in "arnv":
            lemma_param1 = 'n'
        else:
            lemma_param1 = pos1[0]
        if pos2[0] not in "arnv":
            lemma_param2 = 'n'
        else:
            lemma_param2 = pos2[0]

        print(curr[1])
        try:
            word1 = wnl.lemmatize(curr[1].lower(), lemma_param1)
            word2 = wnl.lemmatize(curr[2].lower(), lemma_param2)
            if decade not in decade_unigrams:
                decade_unigrams.update({decade:set([])})
            # add a suffix to the word of the pos, witha space seperator
            word1 = POS_appender(word1, pos1)
            word2 = POS_appender(word2, pos2)
            print(word1,word2)
            if pos1[0:2] != "fo":
                decade_unigrams[decade].add(word1)
            if pos2[0:2] != "fo":
                decade_unigrams[decade].add(word2)
        except UnicodeDecodeError:
            print("Error decoding word")
    return decade_unigrams


def unigram_indices(decade_unigrams):
    index_dict = {}
    for decade in decade_unigrams:
	print(decade)     
        word_indices = {}
        i = 0
        for word in decade_unigrams[decade]:
            word_indices[word] = i
            i += 1
        index_dict[decade] = word_indices
    return index_dict


def bigram_dict(file_name, unigram_dict):
    """ creates a dictioary bi that is a double dictionary. Each inital key is a word from unigram list with a value of another dict, where the subsequent keys would be also be words from the unigram list, with the value being the # of occurrences of the bigram formed by the two keys """
    f = open(file_name, 'r')
    bi = {}
    data = f.readlines()
    f.close()
    for line in data:
        curr = line.split()
        # check both words in unigram list and proceed
        if curr[1] in unigram_dict and curr[2] in unigram_dict:
            if not curr[1] in bi:
                bi.update({curr[1]:{}})
            if not curr[2] in bi[curr[1]]:
                bi[curr[1]].update({curr[2]:1})
            else:
                bi[curr[1]][curr[2]] += 1

    return bi




def construct_historical_matrix(start_year, duration, ngram_file, decade_unigrams):
    data_file = open(ngram_file, "r")
    lines = data_file.readlines()
    data_file.close()
    decade_dict = {}
    index_dict = unigram_indices(decade_unigrams)
    print(index_dict[1800])
    print("1810 as a key should work")
    assert False
    result = []
    numbers = '0123456789'
    
    for line in lines[2:]:
        curr = line.split()
        if len(curr) >= 3:
            word1, word2 = curr[1], curr[2]
            
            # ~~~ lemmatization of word1 word2
            pos1 = curr[3]
            pos2 = curr[4]
            if pos1[0] not in "arnv":
                lemma_param1 = 'n'
            else:
                lemma_param1 = pos1[0]
            
            if pos2[0] not in "arnv":
                lemma_param2 = 'n'
            else:
                lemma_param2 = pos2[0]
            
            word1 = POS_appender(word1, pos1)
            word2 = POS_appender(word2, pos2)

            bigram = (word1, word2)
            freq  = curr[0]

            
    
            if curr[5][0] in numbers:
                
                decade_key = 1800 + 10 * int(curr[5])
  
                if decade_key not in decade_dict and (decade_key <= start_year + duration and decade_key >= start_year):
                    decade_dict[decade_key] = {}
                elif decade_key > start_year + duration or decade_key < start_year:
			        continue
                if word1 not in decade_dict[decade_key]:
                    decade_dict[decade_key][word1] = {}
                decade_dict[decade_key][word1].update({word2:freq})

    # all_matrices contains all co-occurence matrices for each decade, uses a decades as a key
    all_matrix = {}
    print("done adding bigrams")
    f = open("pickle_decade_dict", 'wb')
    pickle.dump(decade_dict, f, -1)
    f.close()
    for decade in decade_dict:
        print(decade)
        length = len(decade_unigrams[decade])
        matrix = dok_matrix((length, length))
        for bigram in decade_dict[decade]:
            word1, word2, freq = bigram[0], bigram[1], decade_dict[decade][bigram]
            matrix[index_dict[decade][word1], index_dict[decade][word2]] = freq
        
        all_matrix.update({decade:matrix})
    print("matrix created")
    return all_matrix

def save_matrix(matrix, decade):
    filename = str(decade)
    matrix = matrix.asformat("csr")
    scipy.sparse.save_npz(filename, matrix)


def co_occurence_matrix(start_year, duration, ngram_file, decade_unigrams):
    wnl = WordNetLemmatizer() 
    data_file = open(ngram_file, "r")
    lines = data_file.readlines()
    data_file.close()
    decade_dict = {}
    index_dict = unigram_indices(decade_unigrams)
    print(index_dict[1810])
    print("1810 as a key should work")
    result = []
    numbers = '0123456789'
    
    for line in lines[2:]:
        curr = line.split()
        if len(curr) >= 3:
            word1, word2 = curr[1], curr[2]
            
            # ~~~ lemmatization of word1 word2
            pos1 = curr[3]
            pos2 = curr[4]
            if pos1[0:2] == 'fo' or pos2[0:2] == 'fo':
                continue
            if pos1[0] not in "arnv":
                lemma_param1 = 'n'
            else:
                lemma_param1 = pos1[0]
            
            if pos2[0] not in "arnv":
                lemma_param2 = 'n'
            else:
                lemma_param2 = pos2[0]
            try:
                word1 = wnl.lemmatize(word1.lower(), lemma_param1)
                word2 = wnl.lemmatize(word2.lower(), lemma_param2)
            except UnicodeDecodeError:
                continue 


            word1 = POS_appender(word1, pos1)
            word2 = POS_appender(word2, pos2)

            bigram = (word1, word2)
            freq  = curr[0]


    
            if curr[5][0] in numbers:
                
                decade_key = 1800 + 10 * int(curr[5])
                if decade_key not in decade_dict and (decade_key <= start_year + duration and decade_key >= start_year):

                    length = len(decade_unigrams[decade_key])                   

                    decade_dict[decade_key] = dok_matrix((length, length))            


                elif decade_key > start_year + duration or decade_key < start_year:
			        continue
                print(word1, word2)

                row = index_dict[decade_key][word1]
                col = index_dict[decade_key][word2]
                decade_dict[decade_key][row,col] += 1
    return decade_dict


def load_matrices(start_year, duration):
    time = start_year
    decade_dict = {}
    while time <= start_year + duration:
        a = dok_matrix(scipy.sparse.load_npz(str(time) + ".npz"))
        decade_dict.update({time: a})
        time += 10
    return decade_dict


def iterate_matrix(decade_dict,start_year, duration, ngram_file, decade_unigrams):
    wnl = WordNetLemmatizer() 
    data_file = open(ngram_file, "r")
    lines = data_file.readlines()
    data_file.close()
    #decade_dict = {}
    index_dict = unigram_indices(decade_unigrams)
    print(index_dict[1810])
    print("1810 as a key should work")
    result = []
    numbers = '0123456789'
    
    for line in lines[2:]:
        curr = line.split()
        if len(curr) >= 3:
            word1, word2 = curr[1], curr[2]
            
            # ~~~ lemmatization of word1 word2
            pos1 = curr[3]
            pos2 = curr[4]
            print("FAIL1")
            if pos1[0:2] == 'fo' or pos2[0:2] == 'fo':
                continue
            if pos1[0] not in "arnv":
                lemma_param1 = 'n'
            else:
                lemma_param1 = pos1[0]
            
            if pos2[0] not in "arnv":
                lemma_param2 = 'n'
            else:
                lemma_param2 = pos2[0]
            try:
                print("fail1.5")
                word1 = wnl.lemmatize(word1.lower(), lemma_param1)
                word2 = wnl.lemmatize(word2.lower(), lemma_param2)
            except UnicodeDecodeError:
                print("EXCEPTION CAUGHT")
                continue 

            print("FAIL2")
            word1 = POS_appender(word1, pos1)
            word2 = POS_appender(word2, pos2)

            bigram = (word1, word2)
            freq  = int( curr[0])


    
            if curr[5][0] in numbers:
                
                decade_key = 1800 + 10 * int(curr[5])
                if decade_key not in decade_dict and (decade_key <= start_year + duration and decade_key >= start_year):

                    length = len(decade_unigrams[decade_key])                   

                    decade_dict[decade_key] = dok_matrix((length, length))            



                elif decade_key > start_year + duration or decade_key < start_year:
			        continue
                print(word1, word2)

                row = index_dict[decade_key][word1]
                col = index_dict[decade_key][word2]
                decade_dict[decade_key][row,col] += freq


    return decade_dict                    
                                                     
    


    
if __name__ == "__main__":  
    #uni = make_unigram_list("unigrams.txt")
    f = open("pickle_unigrams", 'rb')
    uni = pickle.load(f)
    f.close()
    index_dict = unigram_indices(uni)
    f = open("pickl_index_dict", 'wb')
    pickle.dump(index_dict, f)
    f.close()
    #print(uni[1920])
    #print(unigram_indices(uni))
    #print(uni[1810])
    #test_matrix = scipy.sparse.csc_matrix(np.array([[0, 0, 3], [4, 0, 0]]))
    #test1_matrix = test_matrix.todense()
    #np.save("test1.npy", test1_matrix)
    #print(np.load("test1.npy"))
    #print("Done")
    #res = construct_historical_matrix(1810, 170, "bigrams.txt", uni) 
    #res = co_occurence_matrix(1810, 170, "bigrams.txt", uni)i
    #index_dict = unigram_indices(uni) 
    start_year = 1810
    duration = 170
    time = start_year
    decade_dict = {} 
    print(index_dict[1980]["mouse n"])
    assert False
    while time <= start_year + duration:
        length = len(index_dict[time])
        a = dok_matrix((length,length))
        decade_dict.update({time: a})
        time += 10
    
   
    for char in "abcdefghijklmnopqrstuvwxyz":

        decade_dict = load_matrices(start_year, duration)
    
        file_name = char + ".txt"
        res = iterate_matrix(decade_dict, 1810, 170, file_name, uni)
        print(res)
        i = 1
        for decade in res:
            print(i)
            save_matrix(res[decade], decade)
            i+=1 
    
