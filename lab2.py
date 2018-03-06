import scipy
import heapq
import sklearn
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import orthogonal_procrustes
from scipy import spatial
from preproc import *
from lshash import LSHash
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
import pickle

def orthogonal_procrustes(matrix1, matrix2):
    """Transforms matrix2 to matrix1 using orthogonal procrustes. Returns the transformed matrix2"""
    #matrix1 = matrix1 - matrix1.mean(0)
    #matrix2 = matrix2 - matrix2.mean(0)
    print(matrix1.shape, matrix2.shape)
    print(matrix1, matrix2)
    matrix2 = matrix2.transpose()
    print(matrix1.shape, matrix2.shape)
    #M = np.matmul(matrix1, matrix2)
    M = matrix1.dot(matrix2)
    print(M.shape)
    print("M's size is: " + str(M.size))
    u, s, v = scipy.sparse.linalg.svds(M)
    u = scipy.sparse.csr_matrix(u)
    v = scipy.sparse.csr_matrix(v)
    np.save("u", u) #to avoid MemoryErrors
    np.save("v", v)

  

def orthogonal_procrustes_pt2():
    u = np.load("u.npy")
    v = np.load("v.npy")
    #R = u.dot(v)
    R = np.dot(u, v)
    res = matrix2.dot(R)
    return res
    

def align_matrices(matrix1, matrix2):
    """Make matrix1 and matrix2 the same size"""
    changed = 0
    shape1, shape2 = matrix1.shape, matrix2.shape
    len1, len2 = shape1[1], shape2[1]
    diff = len1 - len2
    if diff > 0: #matrix1 is bigger
        #new_matrix = scipy.sparse.csr_matrix((matrix2.data, matrix2.indices, matrix2.indptr), shape=(len2, len2 + diff))

        #cols = np.vsplit(np.asarray(matrix2), len2)
        #arrays = []
       # while diff > 0:
            #arrays.append(np.zeros((len1, 1)))
            #diff -= 1
        #cols.extend(arrays)
        changed = 1
        new_matrix = scipy.sparse.hstack([matrix2, scipy.sparse.csr_matrix((len2, diff))])
    elif diff < 0: #matrix2 is bigger
        #new_matrix = scipy.sparse.csr_matrix((matrix1.data, matrix1.indices, matrix1.indptr), shape=(len1 - diff, len1 - diff))
        #cols = np.split(matrix1.toarray(), len1)
        #arrays = []
        #while diff < 0:
            #arrays.append(np.zeros((len2, 1)))
            #diff -= 1
        #cols.extend(arrays)
        changed = 2
        new_matrix = scipy.sparse.hstack([matrix1, scipy.sparse.csr_matrix((len1, -diff))])
    if changed == 1:
        return (matrix1, new_matrix)
    elif changed == 2:
        return (new_matrix, matrix2)
    return (matrix1, matrix2)

def find_intersections(matrix1, matrix2):
    """Returns the intersection between two matrices"""
    print("")    
    

def k_nearest_neighbours(k, word, decade_matrix, index_dict):
    index_dict = dict(map(reversed, index_dict.items()))
    print(index_dict)  
    k_nearest = []
    nearest_words = {}

    num_rows = decade_matrix.get_shape()[0]
    for i in range(num_rows): 
        comp = decade_matrix.getrow(i).toarray()
        print(i)
        if len(np.nonzero(word)[0]) != 0  and len(np.nonzero(comp)[0]) != 0: #check neither array is all zeroes
            cos_sim = 1 - scipy.spatial.distance.cosine(word, comp[0])
            heapq.heappush(k_nearest, (cos_sim, index_dict[i]))
            print(cos_sim)
                
    k_nearest_words = []
    for item in heapq.nlargest(k, k_nearest):
        k_nearest_words.append(item[1])
    print(k_nearest)
    return (heapq.nlargest(k, k_nearest), k_nearest_words)

def k_nn_lsh(k, word, decade_matrix, index_dict):
    index_dict = dict(map(reversed, index_dict.items()))
    num_rows = decade_matrix.get_shape()[0]
    lsh = LSHash(6, num_rows)
    for i in range(num_rows):
        print(i)
        lsh.index(decade_matrix.getrow(i).todense())
    return lsh.query(word)

def k_nn_lsh_2(k, word, decade_matrix, index_dict):
    num_rows = decade_matrix.get_shape()[0]
    print("the number of rows:" + str(num_rows))
    rbp = RandomBinaryProjections('rbp', 256)
    engine = Engine(num_rows, lshashes=[rbp])
    for i in range(num_rows):
        print(i)
        
        engine.store_vector(decade_matrix.getrow(i), "data_%d" % i)
    return engine.neighbours(word)

def extend_row_vec(vec1, vec2, length, decade_1_index_dict, decade_2_index_dict, smaller):
    new_vec = np.zeros((length,))

    if smaller == "decade1":
        big_index = decade_2_index_dict
        small_index = decade_1_index_dict
        
    else:
        big_index = decade_1_index_dict
        small_index = decade_2_index_dict	
    for key in big_index:
        pos = big_index[key]
        if key in small_index:
            print(small_index[key])
            new_val = vec1[0][small_index[key]]
        else:
            new_val = 0
        #print(new_val)
        print(pos)
        print(new_vec)
        new_vec[pos] = new_val

    return new_vec


def compare_meanings(decade_matrix_1, decade_1_index_dict, decade_matrix_2, decade_2_index_dict, word, k):
    """Compare the similarity of a word's meaning between 2 decades using local neighbourhoods of the vector"""
    
    inv_index_dict_1 = dict(map(reversed, decade_1_index_dict.items()))
    inv_index_dict_2 = dict(map(reversed, decade_2_index_dict.items()))
    
    wv1 = decade_matrix_1.getrow(decade_1_index_dict[word]).toarray()
    wv2 = decade_matrix_2.getrow(decade_2_index_dict[word]).toarray()
    # word_vec_1 = wv1
    #word_vec_2 = wv2
    #smaller = "decade1"
    #if wv1.shape[0] > wv2.shape[0]:
        #word_vec_1 = wv2
        #word_vec_2 = wv1
        #smaller = "decade2"
    
    #word_vec = extend_row_vec(word_vec_1, word_vec_2, max(wv1.shape[1], wv2.shape[1]), decade_1_index_dict, decade_2_index_dict, smaller)
    #print(word_vec)

    decade_1 = k_nearest_neighbours(k, wv1[0], decade_matrix_1, decade_1_index_dict)
    decade_2 = k_nearest_neighbours(k, wv2[0], decade_matrix_2, decade_2_index_dict)
    
    sim_vec_1, sim_vec_2 = decade_1[0], decade_2[0]
    word_list_1, word_list_2 = decade_1[1], decade_2[1]
    decade_1_combined = dict(zip(word_list_1, sim_vec_1))
    decade_2_combined = dict(zip(word_list_2, sim_vec_2))
     
    all_neighbours = list(set().union(word_list_1, word_list_2)) 
    indices = list(range(len(all_neighbours))) #line up vectors
    final_order = dict(zip(all_neighbours, indices))
    final_order_inv = dict(map(reversed, final_order.items()))
    missing_from_1 = [x for x in all_neighbours if x in word_list_2 and x not in word_list_1]
    missing_from_2 = [x for x in all_neighbours if x in word_list_1 and x not in word_list_2]
    
    second_vec_1 = np.zeros(len(all_neighbours))
    second_vec_2 = np.zeros(len(all_neighbours))
    
    for i in range(len(all_neighbours)):
        if final_order_inv[i] in word_list_1:
            second_vec_1[i] = decade_1_combined[final_order_inv[i]][0]
        if final_order_inv[i] in word_list_2:
	    second_vec_2[i] = decade_2_combined[final_order_inv[i]][0]
 
    #fill in missing information for the non-overlapping nearest neighbours
   
    for item in missing_from_1:
        if item in inv_index_dict_1:
            row = decade_matrix_1.getrow(index_dict_1[item])
            cos_sim = 1 - scipy.spatial.distance.cosine(row, word_vec)
            i = final_order[item]
            sim_vec_1[i] = cos_sim
       
    
    for item in missing_from_2:
	if item in inv_index_dict_2:
            row = decade_matrix_2.getrow(inv_index_dict_1[item])
            cos_sim = 1 - scipy.spatial.distance.cosine(row, word_vec)
            i = final_order[item]
            sim_vec_2[i] = cos_sim

    return 1 - scipy.spatial.distance.cosine(second_vec_1, second_vec_2)  


def compare_all_meanings(filename, start_decade, duration, ngram_file, decade_unigrams, k):
    data_file = open(filename, "r")
    lines = data_file.readlines()
    data_file.close()    

    
    matrices = construct_historical_matrix(start_decade, duration, ngram_file, decade_unigrams)
    uni_indices = unigram_indices(decade_unigrams)
    res = []
    
    for line in lines:
        curr = line.split()
        word, word_type = curr[2], curr[3] #TODO: factor in word type
	change = compare_meanings(matrices[start_decade], uni_indices[start_decade], matrices[start_decade + duration], uni_indices[start_decade + duration], word, k)
        res.append((word, change))
    
    return sorted(res, key=lambda x: x[1])


def leaderboards(n, meaning_changes, start_decade, duration, top=True):
    filename = (1 * top) * "most_" + (1 * (not top)) * "least_" + str(n) + "_changed_" + str(start_decade) + "_to_" + str(start_decade + duration)
    f = open(filename, "w")
    if top:
        for i in range(n):
            f.write(str(meaning_changes[i]) + "\n")
    else:
        for i in range(n):
            f.write(str(meaning_changes[::-1][i]) + "\n")
    f.close() 

if __name__ == "__main__":
    #row = np.array([0, 0, 1, 2, 2, 2])
    #col = np.array([0, 2, 2, 0, 1, 2])
    #data = np.array([1, 2, 3, 4, 5, 6])
    #index_dict = {0: "flower", 1: "pot", 2: "bagel"}
    #index_dict_2 = {0: "table", 1: "chair", 2: "flower"}
    #matrix = csr_matrix((data, (row, col)), shape=(3, 3))
    #word = np.array([0, 0, 30])
    #print(k_nearest_neighbours(2, word, matrix, index_dict))
    #print(compare_meanings(matrix, index_dict, matrix, index_dict, "flower", 3)) #should return 1.0
    #print(compare_meanings(matrix, index_dict, matrix, index_dict_2, "flower", 3))
    #assert False
    #leaderboards(1, [("cat", 0), ("dog", 0.3), ("snake", 0.85)], 999, 888, False)
    matrix_1900 = scipy.sparse.load_npz("1900.npz")
    matrix_1980 = scipy.sparse.load_npz("1980.npz")
    #print(scipy.sparse.isspmatrix_csr(matrix_1950))
    #print(scipy.sparse.isspmatrix_csr(matrix_1960))
    matrix_1900, matrix_1980 = align_matrices(matrix_1900, matrix_1980)
    #print(matrix_1950.shape, matrix_1960.shape)
    #R = scipy.linalg.orthogonal_procrustes(matrix_1950, matrix_1960)
    #res = matrix_1950.dot(R)
    #ortho_1960 = orthogonal_procrustes(matrix_1950, matrix_1960)
    #print(ortho_1960)
    #res = orthogonal_procrustes_pt2()
    #res = scipy.spatial.procrustes(matrix_1950, matrix_1960)
    f = open("pickl_index_dict", "r")
    print("gay nearest neighbours")
    index_dict_decade = pickle.load(f)
    gay1 = matrix_1900.getrow(index_dict_decade[1900]["gay a"]).toarray()
    gay11 = k_nearest_neighbours(10, gay1, matrix_1900, index_dict_decade[1900]) 
    print(gay11) 
   
    gay2 = matrix_1980.getrow(index_dict_decade[1980]["gay a"]).toarray()
    gay22 = k_nearest_neighbours(10, gay2, matrix_1980, index_dict_decade[1980]) 
    print(gay22) 
   
    print("mouse nearest neighbours")
    mouse1 = matrix_1900.getrow(index_dict_decade[1900]["mouse n"]).toarray()
    mouse11 = k_nearest_neighbours(10, mouse1, matrix_1900, index_dict_decade[1900]) 
    print(mouse11) 
    
    mouse2 = matrix_1980.getrow(index_dict_decade[1980]["mouse n"]).toarray()
    mouse22 = k_nearest_neighbours(10, mouse2, matrix_1980, index_dict_decade[1980]) 
    print(mouse22)
 
    print("the nearest neighbours")
    the1 = matrix_1900.getrow(index_dict_decade[1900]["the det"]).toarray()
    the11 = k_nearest_neighbours(10, the1, matrix_1900, index_dict_decade[1900]) 
    print(the11) 
    
    the2 = matrix_1980.getrow(index_dict_decade[1980]["mouse n"]).toarray()
    the22 = k_nearest_neighbours(10, the2, matrix_1980, index_dict_decade[1980]) 
    print(the22)
 
    res = compare_meanings(matrix_1900, index_dict_decade[1900], matrix_1980, index_dict_decade[1980], "trained a", 10)
    res2 = compare_meanings(matrix_1900, index_dict_decade[1900], matrix_1980, index_dict_decade[1980], "scrap n", 10)
    res3 =compare_meanings(matrix_1900, index_dict_decade[1900], matrix_1980, index_dict_decade[1980], "slope n", 10)
    res4 = compare_meanings(matrix_1900, index_dict_decade[1900], matrix_1980, index_dict_decade[1980], "occupational a", 10)
    res5 = compare_meanings(matrix_1900, index_dict_decade[1900], matrix_1980, index_dict_decade[1980], "attribute v", 10)
    res6 =compare_meanings(matrix_1900, index_dict_decade[1900], matrix_1980, index_dict_decade[1980], "command n", 10)
    
    print("trained")
    print(res)
  
    trained1 = matrix_1900.getrow(index_dict_decade[1900]["trained a"]).toarray()
    trained11 = k_nearest_neighbours(10, trained1, matrix_1900, index_dict_decade[1900]) 
    print(trained11) 
   
    trained2 = matrix_1980.getrow(index_dict_decade[1980]["trained a"]).toarray()
    trained22 = k_nearest_neighbours(10, trained2, matrix_1980, index_dict_decade[1980]) 
    print(trained22) 
     

    print("scrap")
    print(res2)
 
    scrap1 = matrix_1900.getrow(index_dict_decade[1900]["scrap n"]).toarray()
    scrap11 = k_nearest_neighbours(10, scrap1, matrix_1900, index_dict_decade[1900]) 
    print(scrap11)

    scrap2 = matrix_1980.getrow(index_dict_decade[1980]["scrap n"]).toarray()
    scrap22 = k_nearest_neighbours(10, scrap2, matrix_1980, index_dict_decade[1980]) 
    print(scrap22) 
     

    print("slope")
    print(res3)
  
    slope1 = matrix_1900.getrow(index_dict_decade[1900]["slope n"]).toarray()
    slope11 = k_nearest_neighbours(10, slope1, matrix_1900, index_dict_decade[1900])
    print(slope11) 

    slope2 = matrix_1980.getrow(index_dict_decade[1980]["slope n"]).toarray()
    slope22 = k_nearest_neighbours(10, slope2, matrix_1980, index_dict_decade[1980]) 
    print(slope22) 
     
    print("occupational")
    print(res4)
 
    occ1 = matrix_1900.getrow(index_dict_decade[1900]["occupational a"]).toarray()
    occ11 = k_nearest_neighbours(10, occ1, matrix_1900, index_dict_decade[1900]) 
    print(occ11) 
   
    occ2 = matrix_1980.getrow(index_dict_decade[1980]["occupational a"]).toarray()
    occ22 = k_nearest_neighbours(10, occ2, matrix_1980, index_dict_decade[1980]) 
    print(occ22)
 
    print("attribute")
    print(res5)
    
    att1 = matrix_1900.getrow(index_dict_decade[1900]["attribute v"]).toarray()
    att11 = k_nearest_neighbours(10, att1, matrix_1900, index_dict_decade[1900]) 
    print(att11) 
   
    att2 = matrix_1980.getrow(index_dict_decade[1980]["attribute v"]).toarray()
    att22 = k_nearest_neighbours(10, att2, matrix_1980, index_dict_decade[1980]) 
    print(att22)
 
    print("command")
    print(res6)
    
    com1 = matrix_1900.getrow(index_dict_decade[1900]["command n"]).toarray()
    com11 = k_nearest_neighbours(10, com1, matrix_1900, index_dict_decade[1900]) 
    print(com11) 
   
    com2 = matrix_1980.getrow(index_dict_decade[1980]["command n"]).toarray()
    com22 = k_nearest_neighbours(10, com2, matrix_1980, index_dict_decade[1980]) 
    print(com22) 
   
    #f = open("pickle_unigrams", "rb")
    #unigram_list = pickle.load(f)
    #unigram_index = unigram_indices(unigram_list)
    #comparison = compare_all_meanings("lemma.al", 1950, 0, "bigrams.txt", unigram_list, 10)
    #print(comparison)
    #leaderboards(10, comparison, 1950, 0) 
