#!/usr/bin/python3


""" @package Approach_stat
  This module provide the algorithm for the simulated annealing.
"""

# Needed packages
from distutils.log import error
import numpy as np
import random
import math
from random import seed
import threading
import time
import multiprocessing
from sklearn.cluster import AgglomerativeClustering

from utils import *

# Count the number of thread possible in the computer
thread = multiprocessing.cpu_count()


def d_k(a, b, k):
    """ This function compute the distance between a and b on the coordinate indexed by k.
    @param a The first vector.
    @param b The second vector.
    @return The distance.
    """
    d = 0
    for i in k:
        if a[i] != b[i]:
            d += 1
    return d


# def P_D_k(K, V):
#     for a in V:
#         for b in V:
#             if a != b:
#                 if (d_k(a, b, K) != 0) and (d_k(a, b, K) != len(K)):
#                     return False
#     return True


def Make_partition(Indice_list):
    """ A function for Compute_I.
    """
    I = []
    n = len(Indice_list)
    a_parcourir = [True for _ in range(n)]
    for i in range(n):
        if a_parcourir[i]:
            tmp_K = [i]
            for j in range(i+1, n):
                if a_parcourir[j]:
                    if (Indice_list[i][0] == Indice_list[j][0]) or (Indice_list[i][0] == Indice_list[j][1]):
                        tmp_K.append(j)
                        a_parcourir[j] = False
            I.append(tmp_K)
    return I


def Compute_I(V):
    """ This function returns I a partition of N = {1,..,n}.
    @param V The database.
    @return A partition of N.
    """
    len_D = len(V)
    n = len(V[0])
    Indice_List = []
    for i in range(n):
        vect_tmp_0 = []
        vect_tmp_1 = []
        for j in range(len_D):
            if V[j][i] == 0:
                vect_tmp_0.append(j)
            else:
                vect_tmp_1.append(j)
        Indice_List.append([vect_tmp_0, vect_tmp_1])
    I = Make_partition(Indice_List)
    return I


def Compute_A(I, D, a):
    """ This function compute the matrix A decribe in the paper.
    @param I The partition.
    @param D The database.
    @param a The reference point.
    @return The matrix A.
    """
    A = np.zeros((len(D), len(I)), dtype=int)
    for i in range(len(D)):
        for j in range(len(I)):
            if d_k(a, D[i], I[j]) != 0:
                A[i, j] = -1
            else:
                A[i, j] = 1
    return A


def get_epsilon_0(epsilon, lenD):
    """ This function compute the vector of lenD coordinate with the value epsilon.
    @param epsilon The threshold.
    @param lenD The size of D.
    @return A vector of lenD coordinate with the value epsilon.
    """
    return np.array([[epsilon for _ in range(lenD)]])


def get_da(a, D):
    """ This function compute the vector distance between a point and a database.
    @param a The point.
    @param D The Database.
    @return The vector distance between the point a and the database D.
    """
    da = []
    for b in D:
        da.append(distance_hamming(a, b))
    return np.array([da])


def Compute_N(p, a, I):
    """ This function compute the vector distance between a point and another point on I.
    @param a The point.
    @param p The other point.
    @param I the partition.
    @return The vector distance between the point p and a on I.
    """
    N = []
    for i in I:
        N.append([d_k(a, p, i)])
    return np.array(N)


def simulated_annealing(D, epsilon):
    """Peforms simulated annealing to find a solution
    @param D The database.
    @param epsilon The threshold.
    @return The epsilon master template of D if it has been found.
    """
    initial_temp = 200
    # final_temp = 0.1
    temps = 1
    alpha = 0.001

    current_temp = initial_temp
    I = Compute_I(D)
    moyen = compute_vecteur_moyen(D)
    initial_state = []
    for i in I:
        initial_state.append(d_k(moyen, D[0], i))

    # Start by initializing the current state with the initial state
    current_state = initial_state
    solution = current_state
    cost_init = get_cost(current_state, D, epsilon, I)
    if cost_init == 0:
        return Back_to_F2(solution, D, I)

    while temps < 200000:
        neighbor = random.choice(get_neighbors(current_state, epsilon, D, I))
        # Check if neighbor is best so far
        cost1 = get_cost(current_state, D, epsilon, I)
        cost2 = get_cost(neighbor, D, epsilon, I)
        cost_diff = cost1 - cost2

        # if the new solution is better, accept it
        if cost_diff > 0:
            solution = neighbor
            if cost2 == 0:
                return Back_to_F2(solution, D, I)
        # if the new solution is not better, accept it with a probability of e^(-cost/temp)
        else:
            if random.uniform(0, 1) < math.exp(-cost_diff / current_temp):
                solution = neighbor
        # decrement the temperature
        temps += 1
        # current_temp = 1/math.log(1 + temps*(1/9))
        # current_temp = 200/(1+math.log(1+temps))
        # current_temp = 200*(0.85**temps)
        # current_temp = 13 - math.log(2*temps)
        # current_temp = 200/(1 + temps)
        current_temp -= alpha
    return Back_to_F2(solution, D, I)


def Back_to_F2(solution, D, I):
    """ This function throw back a solution of the simulated annealing in F_2.
    @param solution The solution.
    @param D The database.
    @param I The partition of N for the database.
    @return A vector of lenD coordinate with the value epsilon.
    """
    a = D[0]
    out = [a[i] for i in range(len(a))]
    for i in range(len(solution)):
        index = sample(I[i], solution[i])
        for j in index:
            out[i] ^= 1
    return out


def get_cost(state, D, epsilon, I):
    """Calculates cost of the argument state for your solution.
    @param state The current state of the solution.
    @param D The database.
    @param epsilon The threshold.
    @param I The partition.
    @return The cost of the argument state.
    """
    a = D[0]
    A = Compute_A(I, D, a)
    N = Compute_N(Back_to_F2(state, D, I), a, I)
    epsilon_0 = get_epsilon_0(epsilon, len(D))
    da = get_da(a, D)
    c1 = (np.matmul(A, N)).T
    c2 = epsilon_0-da
    vcost = c2-c1
    cost = -sum([negpart(vcost[0][i]) for i in range(len(vcost))])
    return cost


def get_neighbors(state, epsilon, D, I):
    """ This function returns neighbors of the argument state for your solution.
    @param state The solution current state.
    @param D The database.
    @param I The partition of N for the database.
    @param epsilon The threshold.
    @return The neighbors of the argument state for your solution.
    """

    neighbors_list = []
    # print(I, state, len(I), len(state))
    helper = [state[i] for i in range(len(state))]
    for i in range(len(state)):
        n = min(epsilon, len(I[i]))+1
        tmp = (helper[i]+1) % n
        tmp_vect = [state[j] for j in range(
            i)] + [tmp] + [state[j] for j in range(i+1, len(state))]
        neighbors_list.append(tmp_vect)
        tmp = (tmp-2) % 2
        tmp_vect = [state[j] for j in range(
            i)] + [tmp] + [state[j] for j in range(i+1, len(state))]
        neighbors_list.append(tmp_vect)
    # print("NL=", neighbors_list)
    return neighbors_list


def delete_i_in_list(X, i, label):
    """ This function delete the element indexed by i in the database and in the clusters.
    @param X The database.
    @param i The targeted index.
    @param label The clusters labels of X.
    @return A new list and new labels without the targeted element.
    """
    A = list(X)
    lab = list(label)
    for j in range(len(A)):
        if list_cmp(A[j], i):
            A.pop(j)
            lab.pop(j)
            break
    B = np.array(A)
    labe = np.array(lab)
    return B, labe


def clustering(X, seuil):
    """ This function compute the clusters for a given database and a given threshold.
    @param X The database.
    @param seuil The threshold.
    @return The labels of the clusters and the number of clusters.
    """
    if len(X) <= 1:
        raise error("Pas de cluster sur un seul éléments !")
    Matrix = np.array(compute_HammingDistance_matrix(X))
    clustering = AgglomerativeClustering(n_clusters=None, linkage="complete",
                                         affinity="precomputed", distance_threshold=seuil).fit(Matrix)
    clustering
    label = clustering.labels_
    nbr_clust = max(label)+1
    return label, nbr_clust


########################################################################################
########################################################################################


def Partition(X, epsilon):
    """ This function run the whole attack.
    @param X The database.
    @param seuil The threshold.
    @return The partion of X.
    """
    seuil = 2*epsilon
    label, nbr_clust = clustering(X, seuil)
    part = []
    while len(X) != 0:
        if len(X) == 1:
            nbr_clust = 0
            part.append(X[0])
            break
        else:
            label, nbr_clust = clustering(X, seuil)
        for i in range(nbr_clust):
            my_cluster = compute_cluster(X, label, i)
            if len(my_cluster) == 1:
                sol = True
                solution = my_cluster[0]
            else:
                solution = simulated_annealing(my_cluster, epsilon)
                sol = verification(solution, my_cluster, epsilon)
            # print(sol, epsilon)
            if sol:
                part.append(solution)
                for i in my_cluster:
                    X, label = delete_i_in_list(X, i, label)
        seuil -= 1
    return part, len(part)


def test_partition(n, epsil, nbr_client, sead=123):
    """Set up experiments."""
    templatelist = gen_cli(n, nbr_client, sed=sead)
    return Partition(templatelist, epsil)


def several_iter(n, epsilon, nbr_cli, sed, repet):
    """Set up experiments"""
    nbr_cluster = 0
    temps = 0
    seed(sed)
    global Res_thread
    for _ in range(repet):
        tmps1 = time.perf_counter()
        a, b = test_partition(n, epsilon, nbr_cli, sead=sed)
        tmps2 = time.perf_counter()
        nbr_cluster += b
        temps += (tmps2-tmps1)
    Res_thread.append(np.array([nbr_cluster, temps]))
    return 0


def main_test(nbr_thread, L, rep=1000):
    """"Run and test the whole attack with the given parameters.
    @param nbr_thread The number of thread to use for the tests.
    @param L The vector of parameters.
    """
    global Res_thread
    f = open("Res_global.txt", "w+")
    f.write("$n$ & $\\epsilon $ & $\\#$clients & $\\#$ clust & Time")
    print("$n$ & $\\epsilon $ & $\\#$clients & $\\#$ clust & Time")
    for liste in L:
        Res_thread = []
        n = liste[0]
        epsilon = liste[1]
        nbr_cli = liste[2]
        thread_list = []
        for i in range(nbr_thread):
            thread = threading.Thread(target=several_iter, args=(
                n, epsilon, nbr_cli, random.random()*i, rep//nbr_thread))
            thread.start()
            thread_list.append(thread)
        for t in thread_list:
            t.join()
        k = np.array([0., 0.])
        for i in Res_thread:
            k += i
        nb_cluclu = round(k[0]/rep, 3)
        time = round(k[1]/rep, 3)
        string = "$" + str(n) + "$ & $" + str(epsilon) + "$ & $" + str(nbr_cli) + \
            "$ & $" + str(nb_cluclu) + "$ & $" + str(time) + "$ s \\\\"
        f.write(string)
        print("$", n, "$ & $", epsilon, "$ & $", nbr_cli,
              "$ & $", nb_cluclu, "$ & $", time, "$ s \\\\")
    return 0


def several_iter_marche(n, epsilon, nbr_cli, sed, repet):
    """Set up for test but only for the master-template"""
    temps = 0
    nbr_erreur = 0
    global Res_thread
    for _ in range(repet):
        templatelist = gen_near_templates(
            n, nbr_cli, epsilon, sed=sed*random.random())
        tmps1 = time.perf_counter()
        sol = simulated_annealing(templatelist, epsilon)
        tmps2 = time.perf_counter()
        if not verification(sol, templatelist, epsilon):
            nbr_erreur += 1
        temps += (tmps2-tmps1)

    Res_thread.append([temps, nbr_erreur])
    return True


def main_test_marche_alea(nbr_thread, L, rep=1000):
    """"Run and test the master-template search with the given parameters.
    @param nbr_thread The number of thread to use for the tests.
    @param L The vector of parameters.
    """
    global Res_thread
    f = open("Res_only_sys.txt", "w+")
    f.write("$n$ & $\\epsilon $ & $\\#$clients & $ Error in \% & Time \\\\")
    print("$n$ & $\\epsilon $ & $\\#$clients & $ Error in \% & Time \\\\")
    for liste in L:
        print(liste)
        Res_thread = []
        n = liste[0]
        epsilon = liste[1]
        nbr_cli = liste[2]
        thread_list = []
        for i in range(nbr_thread):
            thread = threading.Thread(target=several_iter_marche, args=(
                n, epsilon, nbr_cli, random.random()*i, rep//nbr_thread))
            thread.start()
            thread_list.append(thread)
        for t in thread_list:
            t.join()
        k = np.array([0., 0.])
        for i in Res_thread:
            k += i
        nb_erreur = round((k[1]/rep)*100, 3)
        time = round(k[0]/rep, 3)
        string = "$" + str(n) + "$ & $" + str(epsilon) + "$ & $" + str(nbr_cli) + \
            "$ & $" + str(nb_erreur) + "$ & $" + str(time) + "$ s \\\\"
        f.write(string)
        print("$", n, "$ & $", epsilon, "$ & $", nbr_cli,
              "$ & $", nb_erreur, "$ & $", time, "$ s \\\\")
    f.close()
    return 0
######################################################
# XP 1
######################################################


# L = []
# for i in range(15, 50, 5):
#     L.append([i, 10, 50])
# 

# L.append([12, 3, 200])
#
# for i in range(5, 40, 5):
#     L.append([70, i, 200])
#
# for i in range(30, 200, 20):
#     L.append([70, 10, i])

# nbr_coeur = 90
# main_test(nbr_coeur, L, rep=nbr_coeur*11)


####################################################
# XP 2 #
####################################################
# w = [i for i in range(50, 70, 5)]
# w = [45]
# for i in w:
#    L.append([i, 10, 50])
#
# main_test_marche_alea(4, L, rep=1000)
