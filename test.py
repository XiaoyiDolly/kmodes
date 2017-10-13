#!/usr/bin/env python

import numpy as np
from kmodes.kmodes import KModes

def multimatch_dissim(a, b, dec_map):
    # print('multimatch_dissim---------------------------->>>>')
    # print(a)
    # print('a',a)
    # print('b',b)
    nattrs = a.shape[1]
    stra = np.empty(a.shape).astype('str') # np.ones([a.shape[0], a.shape[1]], dtype=str)
    strb = np.empty(b.shape).astype('str') # np.ones([b.shape[0], ], dtype=str)
    # print('initial stra------------------>>>>>',stra)
    for iattr in range(nattrs):
        # print(iattr)
        att_str = dec_map[iattr]
        # print (att_str)
        squarer = lambda t: att_str[t]
        vfunc = np.vectorize(squarer)
        strb[iattr] = vfunc(b[iattr])
        stra[:, iattr] = vfunc(a[:, iattr])
        # print('vfuncb------------------->>>>\n', strb[iattr])
        # print('vfunca------------------->>>>\n', stra[:, iattr])

    # print('dis=======================>', np.sum(a!=b, axis=1))
    dis = np.empty([a.shape[0],], dtype=int)
    # print(strb)
    for r,vr in enumerate(stra):
        # print(vr)
        sum = 0
        for c, vc in enumerate(vr):
            # print (c,vc,set(vc.strip().split(' ')))
            # print (strb[c], set(strb[c].strip().split(' ')))
            if not bool(set(vc.strip().split(' ') ) & set(strb[c].strip().split(' '))) or not vc.strip() or not strb[c].strip():
            # if vc.split(' ') not in strb[c] and strb[c] not in vc:
                sum+=1
            # print('sum: ',sum)
        dis[r] = sum
    print(dis)
    return dis

ids = open('data/ids.csv', 'r')
lines = ids.readlines()
id_name = {}
for l in lines:
    l = l.split(' ')
    id_name[l[0]] = l[1]

# reproduce results on small data set
x = np.genfromtxt('test.csv', dtype=str, delimiter=',')[:, 1:]
y = np.genfromtxt('test.csv', dtype=str, delimiter=',', usecols=(0 ))
# x = np.genfromtxt('data/dataset_extract.csv', dtype=str, delimiter=',')[:, 1:]  # test.csv
# y = np.genfromtxt('data/dataset_extract.csv', dtype=str, delimiter=',', usecols=(0 ))
print(y)
print(x.shape)
print(y.shape)
dataNum = y.shape[0]
n_clusters = 10
kmodes_huang = KModes(n_clusters=n_clusters, cat_dissim=multimatch_dissim, init='Huang',n_init=1, verbose=1)
kmodes_huang.fit(x)

# Print cluster centroids of the trained model.
print('k-modes (Huang) centroids:')
print(kmodes_huang.cluster_centroids_)
# Print training statistics
print('Final training cost: {}'.format(kmodes_huang.cost_))
print('Training iterations: {}'.format(kmodes_huang.n_iter_))

# kmodes_cao = KModes(n_clusters=4, init='Cao', verbose=1)
# kmodes_cao.fit(x)
#
# # Print cluster centroids of the trained model.
# print('k-modes (Cao) centroids:')
# print(kmodes_cao.cluster_centroids_)
# # Print training statistics
# print('Final training cost: {}'.format(kmodes_cao.cost_))
# print('Training iterations: {}'.format(kmodes_cao.n_iter_))

print('Results tables:')
for result in (kmodes_huang,): #, kmodes_cao):
    classtable = np.zeros((dataNum,n_clusters), dtype=int)
    for ii, _ in enumerate(y):
        classtable[int(y[ii][-1]) - 1, result.labels_[ii]] += 1

    print("\n")
    print("    | Cl. 1 | Cl. 2 | Cl. 3 | Cl. 4 |")
    print("----|-------|-------|-------|-------|")
    for ii in range(dataNum):
        prargs = tuple([ii + 1] + list(classtable[ii, :]))
        print(" D{0} |    {1:>2} |    {2:>2} |    {3:>2} |    {4:>2} |".format(*prargs))
