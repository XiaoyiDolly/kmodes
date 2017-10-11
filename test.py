#!/usr/bin/env python

import numpy as np
from kmodes.kmodes import KModes

def multimatch_dissim(a, b, dec_map):
    print('multimatch_dissim---------------------------->>>>')
    print(dec_map)
    # print('a',a)
    print('a',a)
    print('b',b)
    nattrs = a.shape[1]
    stra =  np.empty([a.shape[0], a.shape[1]], dtype=str)
    strb = np.empty([b.shape[0], ], dtype=str)
    for iattr in range(nattrs):
        print(iattr)
        att_str = dec_map[iattr]
        print (att_str)
        squarer = lambda t: att_str[t]
        vfunc = np.vectorize(squarer)
        strb[iattr] = vfunc(b[iattr])
        stra[:, iattr] = vfunc(a[:, iattr])
        print('vfuncb------------------->>>>\n',vfunc(b[iattr]))
        print('vfunca------------------->>>>\n', vfunc(a[:, iattr]))
    print('stra------>\n', stra)
    print('strb------>\n', strb)
    print('dis=======================>', np.sum(a!=b, axis=1))
    dis = np.empty([a.shape[0],], dtype=int)
    for r,vr in enumerate(stra):
        print(vr)
        print(strb)
        sum = 0
        for c, vc in enumerate(vr):
            if vc not in strb[c] and strb[c] not in vc:
                sum+=1
        dis[r] = sum
    print(dis)
    return dis


# reproduce results on small soybean data set
x = np.genfromtxt('test.csv', dtype=str, delimiter=',')[:, :-1]
y = np.genfromtxt('test.csv', dtype=str, delimiter=',', usecols=(6, ))
print(x.shape)
kmodes_huang = KModes(n_clusters=4, cat_dissim=multimatch_dissim, init='Huang',n_init=1, verbose=1)
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
    classtable = np.zeros((4, 4), dtype=int)
    for ii, _ in enumerate(y):
        classtable[int(y[ii][-1]) - 1, result.labels_[ii]] += 1

    print("\n")
    print("    | Cl. 1 | Cl. 2 | Cl. 3 | Cl. 4 |")
    print("----|-------|-------|-------|-------|")
    for ii in range(4):
        prargs = tuple([ii + 1] + list(classtable[ii, :]))
        print(" D{0} |    {1:>2} |    {2:>2} |    {3:>2} |    {4:>2} |".format(*prargs))
