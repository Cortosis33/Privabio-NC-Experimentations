#!/usr/bin/python3

""" @package utils
  This module provide some global functions.
  The needed packages are math, numpy, time and random.
"""

# Imports
from math import factorial
import numpy as np
import time
from random import randint
import tqdm
from random import seed


seed(time.time())

"""  Global variable for the timer."""
tmps1 = 0
"""Start the timer."""
tmps2 = 0
"""End of the timer."""


# Functions
def start_timer():
    """"Start the timer: it store the current time in tmps1.
        @return True.
    """
    global tmps1
    tmps1 = time.perf_counter()
    return True


def stop_timer():
    """"End the timer: it store the current time in tmps2.
        @return False.
    """
    global tmps2
    tmps2 = time.perf_counter()
    return False


def get_time():
    """Return the elapsed times between start_timer() and stop_timer().
       @return (tmps2 - tmps1).
    """
    global tmps2, tmps1
    return tmps2 - tmps1


def gen_template(size_of_template, sed=123):
    """ This function allow to generate a binary template of size size_of_template.
    @param size_of_template The size of the wanted template.
    @param sed This is the seed for reproduce some executions.
    @return A vector in {0,1} of lenght size_of_template.
    """
    seed(sed)
    return [randint(0, 1) for _ in range(size_of_template)]


def gen_cli(size_of_template, n, sed=123):
    """ This function allow to generate a whole database: n binary templates of size size_of_template. At the end, each templates are differents and form a numpy array.
    @param size_of_template The size of the wanted template.
    @param n The number of templates wanted in the database.
    @param sed This is the seed for reproduce some executions.
    @return n vector in {0,1} of lenght size_of_template.
    """
    seed(sed)
    L = []
    while len(L) < n:
        A = [randint(0, 1) for _ in range(size_of_template)]
        if not A in L:
            L.append(A)
    return np.array(L)


def gen_DB(size_of_template, n, sed=123):
    """ This function allow to generate a whole database: n binary templates of size size_of_template. At the end, each templates are differents and form a python List.
    @param size_of_template The size of the wanted template.
    @param n The number of templates wanted in the database.
    @param sed This is the seed for reproduce some executions.
    @return n vector in {0,1} of lenght size_of_template.
    """
    seed(sed)
    L = []
    while len(L) < n:
        # print("Client left : ", n-len(L))
        A = [randint(0, 1) for _ in range(size_of_template)]
        if not A in L:
            L.append(A)
    return L


def add_one_template(size_of_template, template_list, sed=123):
    """ This function add a random template in a template list.
    @param size_of_template The size of the wanted template.
    @param template_list The current template list.
    @param sed This is the seed for reproduce some executions.
    @return The inputed list with one more template.
    """
    seed(sed)
    A = [randint(0, 1) for _ in range(size_of_template)]
    # while A in template_list:
    #     A = [randint(0, 1) for _ in range(size_of_template)]
    #     print(A in template_list)
    template_list.append(A)
    return template_list


def pt_in_b(r, n):
    """ This function count the number of template in a ball of radius r in a space of size n.
    @param r The radius of the ball.
    @param n The space size (2**n).
    @return The cardinality of a ball in F_2^n of radius r.
    """
    if r >= n:
        return 2**n
    if r == 0:
        return 1
    if r == 1:
        return n+1
    res = 0
    for i in range(r+1):
        res += factorial(n)/(factorial(n-i)*factorial(i))
    return res


def compute_HammingDistance_matrix(X):
    """ This function compute the dissimilarity matrix of a template database using the Hamming distance.
    @param X The template database.
    @return The dissimilarity matrix.
    """
    n = len(X)
    M = np.zeros((n, n))
    for i in range(n-1):
        for j in range(i+1, n):
            M[i, j] = distance_hamming(X[i], X[j])
            M[j, i] = M[i, j]
    return M


def distance_hamming(A, B):
    """ This function compute the Hamming distance between two templates of same size.
    @param A The first template.
    @param B The second template.
    @return The distance between both template.
    """
    dist = 0
    n = len(A)
    if n != len(B):
        raise "Error vector haven't got the same lenght."
    for i in range(n):
        if A[i] != B[i]:
            dist += 1
    return dist


def opti_ham_dist_tau(A, B, T):
    """ This function say if distance between two template is under a thresold.
    @param A The first template.
    @param B The second template.
    @param T The thresold.
    @return True or False.
    """
    dist = 0
    n = len(A)
    if n != len(B):
        return "Érreur"
    for i in range(n):
        if A[i] != B[i]:
            dist += 1
            if dist >= T:
                return False
    return True


def Add(A, B):
    """ This function return the XOR of two vectors.
    @param A The first template.
    @param B The second template.
    @return  The XOR vector.
    """
    C = []
    if len(A) != len(B):
        raise RuntimeError('Size different in addition vector.')
    for i in range(len(A)):
        C.append(A[i] ^ B[i])
    return C


def gen_mask(alpha, n):
    """
    An helping function for the gen_near_templates method.
    """
    Z = [0 for i in range(n)]
    coord = []
    while len(coord) != alpha:
        a = randint(0, n-1)
        while a in coord:
            a = randint(0, n-1)
        coord.append(a)
    for c in coord:
        Z[c] = 1
    return Z


def gen_near_templates(n, nbr_client, tau, sed):
    """ This function return list of several close templates of size n (under tau).
    @param n The size of templates we want.
    @param nbr_client The number of templates we want to create.
    @param tau The thresold for close templates.
    @param sed This is the seed for reproduce some executions.
    @return  A list of several close templates (under tau).
    """
    seed(sed)
    center = [randint(0, 1) for _ in range(n)]
    L = []
    mask_list = []
    while len(L) < nbr_client:
        alpha = randint(1, tau)
        m = gen_mask(alpha, n)
        while m in mask_list:
            alpha = randint(1, tau)
            m = gen_mask(alpha, n)
        mask_list.append(m)
        v = Add(center, m)
        L.append(v)
    return L


def compute_cluster(Liste_vecteur, cluster_label, cluster_index):
    """ This function return the elements in the cluster cluster_index.
    @param Liste_vecteur The database.
    @param cluster_label The result of the clustering algorithm.
    @param cluster_index The cluster we want to get.
    @return A list with all the element in the cluster cluster_index.
    """
    L = []
    for i in range(len(cluster_label)):
        if cluster_label[i] == cluster_index:
            L.append(Liste_vecteur[i])
    res = np.array(L)
    return res


def compute_max_distance(L, moyen):
    """ This function return the maximum pairwise distance between a database and a vector.
    @param L The database.
    @param moyen The vector.
    @return The maximum pairwise distance between L and moyen.
    """
    dist = []
    for i in L:
        dist.append(distance_hamming(i, moyen))
    # print(dist, max(dist))
    return max(dist)


def compute_center_in_R(template):
    """ This function compute the mean vector in the reel field of a database and return it.
    @param template The database.
    @return The mean vector in the reel field of a database.
    """
    center = []
    n = len(template[0])
    m = len(template)
    for i in range(n):
        med = 0
        for j in range(m):
            med += template[j][i]
        center.append(med/m)
    return center


def compute_vecteur_moyen(L):
    """ This function compute the mean vector of a database and return it.
    @param L The database.
    @return The mean vector of a database.
    """
    moyen = np.array([0 for i in range(len(L[0]))])
    for i in L:
        moyen += i
    moyen = moyen / len(L)
    for i in range(len(moyen)):
        moyen[i] = round(moyen[i])
    return moyen


def negpart(x):
    """ This function compute the negative part of x.
    @param x An integer.
    @return 0 if x > 0 and -x otherwise.
    """
    if x >= 0:
        return 0
    else:
        return -x


def list_cmp(A, B):
    """ This function said if two lists A and B are equal or not.
    @param A The first list.
    @param B The second list.
    @return True if the two lists are equal and false otherwise.
    """
    if len(A) != len(B):
        return False
    for i in range(len(A)):
        if A[i] != B[i]:
            return False
    return True


def verification(res, template, tau):
    """ This function said if res is a master-template of template database.
    @param res A vector.
    @param template The database.
    @param tau The threshold.
    @return True res is a master-template and false otherwise.
    """
    for t in template:
        if distance_hamming(res, t) > tau:
            return False
    return True


def sample(Liste, indice):
    """ This function return a sublist of Liste which contain the elements at the index indice.
    @param Liste The list of elements.
    @param indice The list of wanted index.
    @return A sublist of Liste which contain the elements at the index indice.
    """
    out = []
    for i in range(indice):
        out.append(Liste[i])
    return out


def compute_HammingDistance_matrix(X):
    """ This function compute the dissimilarity matrix for the hamming distance of a database.
    @param X A database.
    @return The matrix of dissimilarity.
    """
    n = len(X)
    M = np.zeros((n, n))
    for i in range(n-1):
        for j in range(i+1, n):
            M[i, j] = distance_hamming(X[i], X[j])
            M[j, i] = M[i, j]
    return M


def mod_gen_cli(size_of_template, n, sed=123):
    """ This function create a database.
    @param size_of_template The size of templates.
    @param n The number of templates.
    @return A list of n templates.
    """
    seed(sed)
    L = []
    pbar = tqdm.tqdm(total=n)
    while len(L) < n:
        A = [randint(0, 1) for _ in range(size_of_template)]
        if not A in L:
            pbar.update(1)
            L.append(A)
    return np.array(L)
