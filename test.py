
#!/usr/bin/env python

import numpy as np
from kmodes.kmodes import KModes

# import sys

# reload(sys)
# sys.setdefaultencoding('utf8')

ids = open('data_names/ids.csv', 'r', encoding='utf8')
lines = ids.readlines()
id_name = {}
for l in lines:
    l = l.split(' ',1)
    id_name[l[0]] = l[1].lower()
    # print(l[0])

def semantic_dismilarity(a, b):
    namea = ''
    nameb = ''
    if 'g' not in a and 'g' not in b:
        lista = list(map(lambda x: id_name[x[1:]].strip() if x else '',a.split(' ')))
        namea = ' '.join(lista)
        listb = list(map(lambda x: id_name[x[1:]].strip() if x else '',b.split(' ')))
        nameb = ' '.join(listb)
    else:
        lista = list(map(lambda x: str.replace(id_name[x[1:]].strip(), '>',' ') if x else '', a.split(' ')))
        namea = ' '.join(lista)
        listb = list(map(lambda x: str.replace(id_name[x[1:]].strip(), '>',' ') if x else '', b.split(' ')))
        nameb = ' '.join(listb)

    if '' in lista or '' in listb:
        # print(1)
        return 1
    else:
        if len(lista) <=len(listb):
            sim = sum(map(lambda x: x in nameb, namea.split(' '))) / len(namea.split(' '))
        else:
            sim = sum(map(lambda x: x in namea, nameb.split(' '))) / len(nameb.split(' '))
        # print(lista, listb, sim)
        return 1-sim


def multimatch_dissim(a, b, dec_map):
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

    dis = np.empty([a.shape[0],], dtype=int)

    for r,vr in enumerate(stra):
        sum = 0
        for c, vc in enumerate(vr):
            valuea = vc.strip()
            valueb = strb[c].strip()
            if not any(i in valuea.split(' ')  for i in valueb.split(' ')) or not valuea or not valueb:
                # print('column a--------------->>>>', valuea)
                # print('column b--------------->>>>', valueb)
                # print(semantic_dismilarity(valuea, valueb))
                sum+= semantic_dismilarity(valuea, valueb)
        dis[r] = sum/4
    print(dis)
    return dis


# reproduce results on small data set
x = np.genfromtxt('data_names/dataset_extract.csv', dtype=str, delimiter=',')[:, 0:]
y = np.genfromtxt('data_names/dataset_extract.csv', dtype=str, delimiter=',', usecols=(0 ))
# x = np.genfromtxt('data/dataset_extract.csv', dtype=str, delimiter=',')[:, 1:]  # test.csv
# y = np.genfromtxt('data/dataset_extract.csv', dtype=str, delimiter=',', usecols=(0 ))
print(y)
print(x.shape)
print(y.shape)
dataNum = y.shape[0]
n_clusters = 100
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

# with open('cluster_results','w') as wf:
np.savetxt('labels.out',kmodes_huang.labels_,  fmt='%i',delimiter=',')
np.savetxt('centroids.out',kmodes_huang.cluster_centroids_, fmt='%s',delimiter=',')

for result in (kmodes_huang,): #, kmodes_cao):
    classtable = np.zeros((dataNum,n_clusters), dtype=int)
    for ii, _ in enumerate(y):

        classtable[int(y[ii][-1]) - 1, result.labels_[ii]] += 1

    print("\n")
    print("    | Cl. 1 | Cl. 2 | Cl. 3 | Cl. 4 |")
    print("----|-------|-------|-------|-------|")
    for ii in range(dataNum):
        prargs = tuple([ii + 1] + list(classtable[ii, :]))
        # print(ii)
        # print(prargs)
        print(" D{0} |    {1:>2} |    {2:>2} |    {3:>2} |    {4:>2} |".format(*prargs))
    # table = np.around(classtable[ii, :], decimals=2)
    # np.savetxt('test.out', np.around(classtable[ii, :], decimals=2), delimiter=',')
    # with open('cluster_results','w') as wf:
    #     for ii in range(dataNum):
    #         [ii + 1] + list(classtable[ii, :])
    #         wf.write(classtable[ii, :])
    #         wf.write('\n')