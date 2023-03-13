#!/usr/bin/python3

""" @package Partition
  This module provide the algorithm tests of the whole attack for the IP approach.
"""

# Packages
import numpy as np
import time
from random import random
from random import seed
import tqdm
import multiprocessing

from utils import *
from center import *
from Approach_stat import *

seed(time.time())


thread = multiprocessing.cpu_count()


def delete_i_in_matrix(X, i):
    """"Remove the i-th element of the dissimililarity matrix.
    @param X The dissimilarity matrix.
    @param i The index to remove.
    @return The matrix without i.
    """
    A = list(X)
    for j in range(len(A)):
        if list_cmp(A[j], i):
            A.pop(j)
            break
    B = np.array(A)
    return B


def print_partition(L):
    """"Debug Function print a partition to check if it works.
    @param L The partition.
    """
    my_bool = True
    for i in L:
        print(i)
        if i == "Échec":
            my_bool = False
    print("\n")
    return my_bool


def gluttony(L, epsilon):
    """"The gluttony algorithm to do the partitionning.
    @param L The database.
    @param epsilon The threshold.
    @return The partition.
    """
    clust = []
    while len(L) != 0:
        clust.append(L[0])
        p = L[0]
        L = delete_i_in_matrix(L, L[0])
        for i in L:
            del_lis = []
            if distance_hamming(p, i) < epsilon:
                del_lis.append(i)
        for v in del_lis:
            L = delete_i_in_matrix(L, v)
    return clust


Res_thread = []

Res_glout = []


def test_gouton(n, epsilon, nbr_cli, sezd, repet):
    """Set up to test the gluttony algorithm."""
    nbr_cluster = 0
    temps = 0
    global Res_glout
    for i in tqdm.tqdm(range(repet)):
        L = gen_cli(n, nbr_cli, sed=sezd*i)
        tmps1 = time.perf_counter()
        B = gluttony(L, epsilon)
        tmps2 = time.perf_counter()
        nbr_cluster += len(B)
        temps += (tmps2-tmps1)
    Res_glout.append(np.array([nbr_cluster, temps]))
    return True


def partition2(X, epsilon):
    """The whole attack using the IP algorithm
    @param X The database.
    @param epsilon The threshold
    @return The new database.
    """
    seuil = 2*epsilon
    part = []
    while len(X) != 0:
        label, nbr_clust = clustering(X, seuil)
        for i in range(nbr_clust):
            my_cluster = compute_cluster(X, label, i)
            if len(my_cluster) == 1:
                res = my_cluster[0]
                verif = True
            else:
                res, verif = Find_eps_cov_temp(my_cluster, epsilon)
            if not verif:
                pass
            else:
                part.append(res)
                for i in my_cluster:
                    X, label = delete_i_in_list(X, i, label)
        seuil -= 1
    return part, len(part)


def test_partition2(n, epsil, nbr_client, sead=123):
    """Set up for the experimentations"""
    templatelist = mod_gen_cli(n, nbr_client, sed=sead)
    return partition2(templatelist, epsil)


Res_thread = []


def several_iter2(n, epsilon, nbr_cli, sed, repet):
    """Set up for the experimentations"""
    nbr_cluster = 0
    temps = 0
    global Res_thread
    for _ in range(repet):
        templatelist = mod_gen_cli(n, nbr_cli, sed=sed*random())
        tmps1 = time.perf_counter()
        # d = gluttony(templatelist, epsilon)
        c, d = partition2(templatelist, epsilon)
        tmps2 = time.perf_counter()
        #nbr_cluster += len(d)
        nbr_cluster += d
        temps += (tmps2-tmps1)
    Res_thread.append(np.array([nbr_cluster, temps]))
    return True


def main_test_mt(L, rep=10000):
    """Run the main experimentation using some parameters.
    @param L A list of parameters
    """
    global thread, Res_thread, Res_glout
    tmpslst = []
    Gres = []
    tmp = random()
    for liste in L:
        tmp *= random()
        Res_thread = []
        n = liste[0]
        epsilon = liste[1]
        nbr_cli = liste[2]
        several_iter2(n, epsilon, nbr_cli, random()*tmp, rep)
        k = np.array([0., 0.])
        for i in Res_thread:
            k += i

        nb_cluclu = round(k[0]/rep, 3)
        Gres.append([n, epsilon, nbr_cli, nb_cluclu,
                     round((k[1]/rep)*(10**3), 3)])
        print(Res_thread, Gres)

    print("$n$ & $\\epsilon $ & $\\#$clients & $\\#$ clust & Time")
    for liste in Gres:
        print("$", liste[0], "$ & $", liste[1], "$ & $", liste[2], "$ & $",
              liste[3], "$ & $", liste[4], "$\n")
    return tmpslst


# L = []
# for i in range(15, 50, 5):
#    L.append([i, 10, 50])
#L.append([70, 3, 200])
# for i in range(5, 40, 5):
# for i in range(30, 40, 5):
#     L.append([70, i, 200])
# for i in range(30, 200, 20):
#     L.append([70, 10, i])
#
# main_test_mt(L, rep=100)


# X = []
# n = 8
# for i in itertools.product([0, 1], repeat=n):
#     k = list(i)
#     X.append(k)
# Y = np.array(X)
# e = 2
# part, nbr = partition2(Y, e)
# # print("Partition is :", np.array(part))
# print("The size of the part is :", nbr)
# print("Taille avec glouton : ", len(gluttony(Y, e)))
# print("nbr de cluster optimal :", 2**n/pt_in_b(e, n))
# 
