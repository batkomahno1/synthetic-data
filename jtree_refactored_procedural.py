#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:26:58 2020

@author: kyrylo
"""
import pandas as pd
import numpy as np
import warnings
import itertools
import multiprocessing as mp
from functools import partial
import ipyparallel as ipp
import matplotlib.pyplot as plt
import os
import sys

def getSensitivityAndEpsilon(data, domains, epsilon):
    # TODO: same thing, not sure if domains should be from sampled D
    n = data.shape[0]
    epsilon_a = (np.log(np.exp(epsilon)-1+n/n)-np.log(n/n))
    sensitivity = (np.log10(n)/n + np.log10(n/(n-1))*(n-1)/n)
    
    for col in data.columns:
        if len(domains[col]) > 2:
            sensitivity = (np.log10((n+1)/2)*2/n + np.log10((n+1)/(n-1))*(n-1)/n)
        
    return sensitivity, epsilon_a

def getSamplingRate(data, domains, epsilon):
    data_size = data.shape[0]
    
    sensitivity = lambda n: (np.log10(n)/n + np.log10(n/(n-1))*(n-1)/n)
    epsilon_a = lambda n: (np.log(np.exp(epsilon)-1+n/data_size)-np.log(n/data_size))
    
    # TODO: remove unnecessary iteration
    for col in data.columns:
        if len(domains[col]) > 2:
            sensitivity = lambda n: (np.log10((n+1)/2)*2/n + np.log10((n+1)/(n-1))*(n-1)/n)
    
    # TODO: WHY DID I ADD 2 HERE????
    n_s = np.argmin([sensitivity(i)/epsilon_a(i) for i in range(2, data_size+1)])#+2
    
    return n_s/data_size

def toInclude(data_sampled, domains, noise_scale, pair):
    import numpy as np
    attrib_k, attrib_l = pair[0], pair[1]
    domain_size_k = len(domains[attrib_k])
    domain_size_l = len(domains[attrib_l])
    phi = 0.2 # room to play
    threshold = min(domain_size_k-1,domain_size_l-1)*(phi**2)/2
    mutInfo = 0

    #sklearn implementation
    from sklearn.metrics import mutual_info_score
    mutInfo = mutual_info_score(data_sampled[attrib_k], 
                                data_sampled[attrib_l])

    # TODO: small differences between my implementation and library    
    # TODO: should I generate nosie each time for the threshold?
    noise1=np.random.laplace(scale=noise_scale)
    noise2=np.random.laplace(scale=noise_scale)

    return pair if mutInfo + noise1 >= threshold + noise2 else None


# Forming Attribute Clusters
def buildJunctionTree(G, verbose):
    from pgmpy import models
    import networkx as nx
    import matplotlib.pyplot as plt
    
    if G.number_of_edges() == 0: 
        return [tuple(set(G.nodes()))], set(), G
    
    # check triangulation (if problems arise)
    BM = models.MarkovModel(G.edges())
    G=BM.to_junction_tree()

    if verbose:
        # plot the graph
        nx.draw(G, with_labels=True)
        l,r = plt.xlim()
        plt.xlim(l-2,r+2)
        plt.title('Junction Tree')
        plt.show()

    cliques = list(set(G.nodes()))
    seperators = {tuple(set(e[0]) & set(e[1])) for e in G.edges()}
    
    return cliques, seperators, G

# Optimization Problem
def identifyOptimalMerging(A, domains, C, m):
    # TODO: make glob vars
    #TODO: not sure if domains should be from sampled D??
    import cvxpy as cvx
    import numpy as np
    
    # check boundary conditions
    if m==1: return [set(A)], np.array([0]*len(C))
    # TODO: return m=|C| unmodified? Each clique is a cluster?
    if m==len(C): return [set(clique) for clique in C], np.array(list(range(len(C))))    

    c = {attr: len(domains[attr]) for attr in A}
    d = len(A) # total num of attribs
    n = len(C) # num of cliques
    
    O = np.zeros(shape=(d, n))
    # init occurence matrix. Attrib i in j_th clique
    for i in range(d):
        for j in range(n):
            O[i][j] = 1 if A[i] in C[j] else 0
    
    p = np.zeros(len(C))
    # init p; prod of attrib cardinalities in a clique
    for i in range(n):
        p[i] = np.product([c[attr] for attr in C[i]])
    

    # find local min guaranteed by the CCCP method
    opt_solns = np.array([-np.inf, np.inf, -np.inf])
    
    # set initial value to random as per source below:
    # A. J. Smola, S. V. N. Vishwanathan, and T. Hofmann, “Kernel Methods for Missing Variables,” p. 8.
    # TODO: should initial values add up to 1 column-wise? As per prob. theory?
    Z_prev = np.random.uniform(low=0.0, high=1.0, size = (n,m)) 
    
    while not (opt_solns[0]>opt_solns[1]<opt_solns[2]):        
        # set variables, constants, etc
        t = cvx.Variable(m, nonneg=True) # special expression
        Z = cvx.Variable((n,m), nonneg=True) # Z[i][k] -> i_th clique in k_th cluster
        P = np.empty(shape=(m, m), dtype=object) # Z[:,i] - Z[:,j] diff btwn cols of Z
        Q = np.empty(shape=(m, m), dtype=object) # ||P[i][j]||**2
        
        # init P and Q
        for row in range(m):
            for col in range(m):
                P[row][col] = Z_prev[:,row] - Z_prev[:,col]
                # TODO: check elementwise exp
                Q[row][col] = np.linalg.norm(P[row][col], 2)**2

        # set lambda and r
        # TODO: better choices for lambda!
        lda = 0.005 #cvx.Parameter(nonneg=True, value = 0.01)
        r = cvx.Variable(1, nonneg=True)

        # objective
        # TODO: leave out lambda*r? Part of cvxpy classs?
        objective = cvx.Minimize(cvx.log_sum_exp(t) - lda*r)

        # constraints
        const1 = []
        for row in range(n):
            const1 += [cvx.abs(cvx.sum(Z[row, :]) - 1.0) <= 1e-15] # TODO: floating point arithmetic!

        const2 = []
        for row in range(m):
            for col in range(m):
                if row != col:
                    const2 += [r - 2*(Z[:,row] - Z[:,col]).T@P[row][col] + Q[row][col] <= 0]
        
        const3 = []
        for col in range(m):
            const3 += [cvx.sum(Z[:, col]) >= 1.0]
            
        const4 = []
        for k in range(m):
            const4 += [
                # TODO: make sure the -1 in first inner sum belongs outside the sum !
                cvx.sum([val[1]*np.log(p[val[0]]) for val in \
                         enumerate(Z[:,k])]) - t[k] - cvx.sum([(cvx.sum([Z[j][k]*O[i,j] for j in \
                                                                         range(n)])-1)*np.log(c[i]) for i in range(d)]) + \
                np.log(np.sum(Z_prev[:,k])) + \
                cvx.sum(Z[:,k]-Z_prev[:,k])/np.sum(Z_prev[:,k]) <= 0
            ]
        
        constraints = const1+const2+const3+const4
        
        problem = cvx.Problem(objective, constraints)
        
        problem.solve(solver=cvx.SCS)
        
        opt_solns[0], opt_solns[1], opt_solns[2] = opt_solns[1], opt_solns[2], problem.value
        
        Z_prev = Z.value
        
    if Z.value is None: return problem.status
    
    Z_prob = Z_prev
    Z_det = np.zeros(shape=(n,m))
    
    for coord in zip(range(Z_prob.shape[0]), Z_prob.argmax(axis=1)):
        Z_det[coord]=1            
        
    assert all([1 in row for row in Z_det]), ['Not all cliques assigned to clusters!']
    
    clusters=[()]*m
    clique_address = np.where(Z_det==1)[1]

    for i in range(len(C)):
        clusters[clique_address[i]] = clusters[clique_address[i]]+C[i]
        
    assert len(clique_address) == len(C), Z_det
    
    return [set(c) for c in clusters], clique_address

def calcTotNoiseVar(domains, epsilon, clusters):
    import numpy as np
    m = len(clusters)
    
    noiseVar = 0
    for i in range(0, m):
        # m+1 bc counting from zero
        noiseVar += (8*(m+1)**2/epsilon**2)*len(clusters[i])* \
                    np.product([len(domains[attrib]) for attrib in clusters[i]])
    return clusters, noiseVar

def chooseClusterSets(clusters):
    if len(clusters)==0: return dict()
    
    import itertools
    combos = [j for i in [list(itertools.combinations(clusters,n)) for n in range(2, len(clusters)+1)] for j in i]
    intersections={tuple(tuple(v) for v in list(combo)):tuple(set.intersection(*combo)) for combo in combos}
    
    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from([tuple(v) for v in list(set(intersections.values()))])

    intersections_sorted = list(reversed(list(nx.topological_sort(G))))
    
    cluster_sets = dict()
    for val in intersections_sorted:
        var = {k:v for k,v in intersections.items() if tuple(v) == val}
        cluster_sets[max(var.keys(), key = len)]=set(val)
        
    return cluster_sets

class JTree:
    def __init__(self, data: pd.DataFrame, table_size=0, domains=None, verbose=False, epsilon=1.0):
        # silence prints if needed
        self.verbose = verbose
        
        # TODO: check data types
        self.data = data
        self.attributes = self.data.columns
        
        # TODO: should set domain properly
        if domains is None:
            warnings.warn("Domains not provided. Privacy leak!", UserWarning)
            self.domains = {a:set(self.data[a].values) for a in self.attributes}
        else:
            self.domains = domains

        # per paper
        self.epsilon = epsilon
                
        # size of the output table
        self.table_size= table_size if table_size != 0 else self.data.shape[0]
        
        # this is where the result is stored
        self.table = pd.DataFrame(np.NaN, index=range(self.table_size), columns=self.attributes)
        
        # set multiprocessing engine
        import ipyparallel as ipp
        client = ipp.Client()
        print(f'Running on {client.ids} cores')
        self.view = client[:]
        
        # clear engines from junk
        self.view.client.purge_everything()
        
        if self.verbose: print(f'Changing engines CWD to {os.getcwd()}')
        newdirs = [os.getcwd()]*len(self.view)
        self.view.map(os.chdir, newdirs)
        assert self.view.apply_sync(os.getcwd) == newdirs

    def generateDepGraph(self, epsilon):
        B=getSamplingRate(self.data, self.domains, epsilon)
        data_sampled = self.data.sample(frac=B) # sampling prob could cause problems
        # TODO: use sample D domains or global? Leave the option open.
        sensitivity, epsilon_a = getSensitivityAndEpsilon(data_sampled,
                                                          self.domains, epsilon)
        noise_scale = 2*sensitivity/epsilon_a
        
        # start building graph
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(self.attributes)
        attrib_pairs = list(nx.non_edges(G))
        
        # parallel version        
        # TODO: use sample D domains or global? Leave the option open.
        
        var = self.view.map_sync(partial(toInclude, data_sampled, self.domains, noise_scale), attrib_pairs)
                
        G.add_edges_from(filter(lambda x: x is not None, var))
        
        # draw Graph
        if self.verbose:
            nx.draw(G, with_labels=True)
            plt.title('Dependency Graph')
            plt.show()
        
        return [G.subgraph(c) for c in nx.connected_components(G)]
        
    def formAttributeClusters(self, C, S, attributes, domains, epsilon):
        # TODO: make glob vars
        if len(C)==1: 
    #         assert isinstance(C[0], tuple), C
            return C, None, [set(C[0])], np.array([0])
        
        # TODO: check DP on parallel version!
        # parallel version
        var = self.view.map_sync(partial(identifyOptimalMerging, attributes, domains, C), range(1, len(C)+1))

        if self.verbose: print(f"Processing {len(C)} possible mergins...")
        
        # remove failed mergings
        mergings=[]
        for i, merging in enumerate(var):
            if not isinstance(merging[0], str):
                mergings.append(merging[0])
            else:
                # TODO: throw a warning here
                if self.verbose: print(f"Merging m = {i+1} is {merging}")
#                 raise Exception('Optimization Failed')
                
        if self.verbose: print(f'Mergings are {mergings}')
        noiseVars = self.view.map_sync(partial(calcTotNoiseVar, domains, epsilon), mergings)
    
        selectedMerging=([], np.inf)
        for val in noiseVars:
            if val[1] < selectedMerging[1]: selectedMerging=val 
    
        return C, S, selectedMerging[0], [v[1] for v in var if v[0]==selectedMerging[0]][0]

    def makeNoisyMarginals(self, clusters, epsilon):
        # Marginal tables should be based on original data
        # set variables
        m = len(clusters)
        D_size = self.data.shape[0]
        
        if self.verbose: print(f'Adding noise to clusters: {clusters}')
        
        # form clean cluster marginals
        T_CL = {tuple(cluster): self.data.groupby(list(cluster)).size().reset_index(name='margin') \
                for cluster in clusters}
        
        assert all([tcl['margin'].sum() == D_size for tcl in T_CL.values()]), [T_CL]
        
        # TODO: noise seems insufficient!
        # add noise
        for c in T_CL.keys():
            noise = np.random.laplace(scale=2*m/epsilon)
            T_CL[c]['margin'] += noise
        
        return T_CL

    def makeMarginalsConsistent(self, clusters, A, T_CL, domains):
        # TODO: make glob vars
        # don't enforce consistency if intersection is empty
        if len(A) == 0: return T_CL    
                    
        # create a set of possible domain values in A
        dom_A = list(itertools.product(*[domains[a] for a in A]))
        
        # PARALLELIZE THIS!!!
        
        # adjust marginal tables
        for idx, a in enumerate(dom_A):
            # approximate T_A
            if idx%(10**(len(str(len(dom_A)))-1)/100)==0:
                if self.verbose: print(f'\rUpdating attribtue {idx} out of {len(dom_A)}', end='')
                
            num, denom = 0.0, 0.0
            for c in T_CL.keys():
                if set(A).issubset(c):
                    # TODO: is summing margins of attrib subset from bigger cluster the right solution here?
                    var1 = T_CL[c][(T_CL[c][A] == a).all(1)]['margin'].sum()
                    var2 = np.product([len(domains[attrib]) for attrib in set(c) - A])            
                    num += var1/var2            
                    denom += 1/np.product([len(domains[attrib]) for attrib in set(c) - A])            
            T_A = num/denom
            
            # update all T_CLs to be consistent with T_A
            for c in T_CL.keys():
                if set(A).issubset(c):
                    var1 = T_CL[c][(T_CL[c][A] == a).all(1)]['margin'].sum()
                    var2 = np.product([len(domains[attrib]) for attrib in set(c) - A])
                    T_CL[c]['margin'] += (T_A - var1) / var2
    
        return T_CL    

    def mitigateBias(self, T_CL, D_size):
        # TODO: make glob vars
        # TODO: SOLVE THIS PROPERLY !!
        # D_size should be based on original data
        # mitigate bias introduced by Laplace variables
        for c in T_CL.keys():
            var= int(np.floor(T_CL[c]['margin'].min()))
            min_ = var if var > 0 else 1
            max_ = int(np.ceil(T_CL[c]['margin'].max()))
            tholds = list(range(min_, max_+1))
    
            if T_CL[c]['margin'].max() < 1: tholds = np.linspace(0.1,1, 10)
                
            assert 0 not in tholds, [tholds] 
            
            # TODO: parallelize this
            N_dict = {thold:T_CL[c]['margin'][T_CL[c]['margin'] >= thold].sum() for thold in tholds}
            
            assert len(N_dict) > 0, T_CL[c] 
            
            thold, summation = min(N_dict.items(), key=lambda kv: abs(kv[1] - D_size))
            
            if self.verbose: print(f'Bias threshold is {thold}')
    
            T_CL[c].loc[T_CL[c]['margin'] > thold, 'margin'] *= D_size/thold
            T_CL[c].loc[T_CL[c]['margin'] <= thold, 'margin'] = 0
        
        return T_CL    

    def buildTable(self, attributes, trees):                                    
        for tree in trees:
            tree_marginals, J = tree
            
            assert all([node in tree_marginals.keys() for node in J.nodes() \
                        if not isinstance(node, int)]), [node for node in J.nodes() \
                        if node not in tree_marginals.keys()]
            
            # calculate marginal frequencies for all cliques(nodes)
            for k in tree_marginals.keys():
                tree_marginals[k]['margin'] /= tree_marginals[k]['margin'].sum()
    
            # safely choose random node
            import secrets
            curr_node = tuple(secrets.choice(list(tree_marginals.keys())))
            
            if self.verbose: print(f'Synthesizing from tree {J.nodes()} and initial node is {curr_node}')
    
            # fill up the initial node
            for _, row in tree_marginals[curr_node].iterrows():
                freq, vals=row.values[-1], row.values[:-1]
                
                assert all(val in self.domains[a] for a, val in zip(list(curr_node), vals))
                
                # get available rows
                available = self.table[self.table[list(curr_node)].isna().all(1)]
                
                # need to sample X nans s.t. X/|D| = freq => X = freq x |D|
                # check if enough entries for this node are avaiable
                if available.shape[0] >= round(freq*self.table.shape[0]):
                    idx = available.sample(int(round(freq*self.table.shape[0]))).index
                else:
                    idx = available.index
                
                assert len(vals)==self.table.loc[idx, list(curr_node)].shape[1], \
                [vals, curr_node, self.table.loc[idx, list(curr_node)], tree_marginals[curr_node]]
                
                self.table.loc[idx, list(curr_node)] = vals
            
#             # drop NaNs
#             nans = self.table[pd.isnull(self.table[list(curr_node)]).any(axis=1)].index
#             if self.verbose: print(f'Dropping NaN rows {nans}')
#             self.table = self.table.drop(nans).reset_index(drop=True).copy()
#             assert not self.table[list(curr_node)].isna().any().any() \
#                                         and self.table.shape[0] > 0, [self.table]
    
            # fill up the rest
            if len(J.nodes()) > 1: self.iterTree(tree_marginals, J, curr_node)
            
        return self.table

    def iterTree(self, tree_marginals, J, curr_node, filled=None, visited=None):
        # default values get stuck in memory !!!
        if filled is None: filled=set()
        if visited is None: visited=set()
            
        if self.verbose: print(f'Processed node {curr_node}')
        
        #get master node's adjecent nodes
        adj = list(J.neighbors(curr_node))
    
        # mark this node visited
        visited.add(curr_node)
        filled.add(curr_node)
        
        # TODO PARALLELIZE THIS
        # this recursion will stop once all visited
        for node in adj:
            if node not in filled:# or True:
                inter = list(set(node).intersection(set(curr_node)))
                cond_vars = list(set(node)-set(inter))

                if self.verbose: print(f'Populating node {node}, variables {cond_vars}, visited {visited}')

                assert len(inter) != 0, [inter, node, curr_node]

                for v in {tuple(v) for v in self.table[inter].values}:
                    MR = tree_marginals[node].copy()
                    for _, row in MR[(MR[inter] == list(v)).all(1)].iterrows():

                        freq, vals = row.loc[['margin']].values[-1], row.loc[cond_vars].values

                        # find intersection in the table
                        df = self.table[(self.table[inter]==list(v)).all(1)]#.copy()

                        # find not filled vals in the intersection
                        available = df[pd.isnull(df[cond_vars]).any(axis=1)]
                        
                        if available.shape[0] >= round(freq*self.table.shape[0]):
                            idx = available.sample(int(round(freq*self.table.shape[0]))).index
                        else:
                            idx = available.index

                        # fill
                        self.table.loc[idx, cond_vars] = vals
                filled.add(node)
        for node in adj:
            if node not in visited:
                # recurse over its neighbors
                self.iterTree(tree_marginals, J, node, filled, visited)

    def synthesize(self):
        import time
        start = time.time()
        
        # per paper
        epsilon_1 = 0.1
        
        # TODO: process each tree seperately!!
        # espilon 1 used here
        trees = self.generateDepGraph(epsilon_1)
        
        #TODO: DOUBLE CHECK CORRECTNESS OF THE EPS COMPOSITION !!
        epsilon_2 = (self.epsilon - epsilon_1)#/len(trees)
        
        tree_marginals = []
        for G in trees:
            if self.verbose: print(f'Processing tree {G.nodes()}')
            
            C, S, J = buildJunctionTree(G, self.verbose)
            
            # the output below is noise-free; epsilon is used to calculate noise var
            # espilon 2 used here
            C, S, clusters, clique_address = self.formAttributeClusters(C, S, self.attributes, 
                                                                        self.domains, epsilon_2)
            
            assert all([node in C for node in J.nodes() if not isinstance(node, int)]), \
            [node for node in J.nodes() if node not in C]
    
            '''
            TODO: how do we split epsilon between tree? 
            1) Divide it by the number of trees OR 
            2) set m=total number of clusters when adding noise OR 
            3) leave epsilon in tact and divide by number of clusters on per tree basis
            '''        
            # add noise to marginals
            # espilon 2 used here
            marginals = self.makeNoisyMarginals(clusters, epsilon_2)#/len(trees))
    
            # get sets of clusters and their intersections
            cluster_sets = chooseClusterSets(clusters)
    
            # TODO: check consistency programatically!
            # make margnials consistent
            for CL, A in cluster_sets.items():
                marginals = self.makeMarginalsConsistent(CL, A, marginals, self.domains)
    
            # remove negative vals and other garbage from Laplacian nosie
            marginals = self.mitigateBias(marginals, self.data.shape[0])
    
            clique_marginals=dict()
            for idx, clique in enumerate(C):
                
                cluster = tuple(clusters[clique_address[idx]])
                
                assert isinstance(clique, tuple)
                
                if self.verbose: print(f'Calculating marginals of clique {clique} from {cluster}')
                
                clique_marginals[clique] = marginals[cluster][list(clique)+['margin']].\
                                                    groupby(list(clique)).sum().reset_index()
        
                assert set(clique_marginals[clique].columns[:-1]) == set(clique), \
                                        [set(clique_marginals[clique].columns[:-1]), set(clique)]
            
            tree_marginals.append((clique_marginals, J))
            
        # TODO: What is max table size?
        synth_table = self.buildTable(self.attributes, tree_marginals)
        
        print(f"\nSynthesis execution time: {round(time.time() - start, 0)} s\n")
        
        return synth_table, tree_marginals
