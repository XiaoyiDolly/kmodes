
#!/usr/bin/env python

import numpy as np
from kmodes.kmodes import KModes

# import sys

# reload(sys)
# sys.setdefaultencoding('utf8')


def init(cat='n&c'):
    if cat =='n&c':
        ids = open('data_category/ids.csv', 'r', encoding='utf8')
        lines = ids.readlines()
        id_name = {}
        id_cat = {}
        for l in lines:
            l = l.split(';')

            if len(l) ==3:
                id_cat[l[0]] = l[2].split('>')
                id_name[l[0]] = l[1].lower()
            else:
                id_cat[l[0]] = l[1].split('>')
                id_name[l[0]] = l[1].split('>')[-1].lower()
    elif cat == 'n':
        ids = open('data_names/ids.csv', 'r', encoding='utf8')
        lines = ids.readlines()
        id_name = {}
        for l in lines:
            l = l.split(' ',1)
            id_name[l[0]] = l[1].lower()
    return id_name, id_cat

ids = open('data_category/ids.csv', 'r', encoding='utf8')
lines = ids.readlines()
id_name = {}
id_cat = {}
for l in lines:
    l = l.strip().split(';')
    # print(l)
    if len(l) ==3:

        id_cat[l[0]] = l[2].split('>')
        id_name[l[0]] = l[1].lower()
    else:
        # print('2222222', l)
        id_cat[l[0]] = l[1].split('>')
        id_name[l[0]] = l[1].split('>')[-1].lower()
    if l[0] ==77:
        print('*********',l)

print (id_cat['0'])
print (id_cat['82'])
print (id_cat['6543'])
print (id_cat['143'])

def semantic_disimilarity(a, b):
    print(a, b)
    if not a or not b:
        # print(1)
        return 1
    namea = ''
    nameb = ''
    # if 'g' not in a and 'g' not in b:
    lista = list(map(lambda x: id_name[x[1:]].strip() if x else '',a.split(' ')))
    namea = ' '.join(lista)
    listb = list(map(lambda x: id_name[x[1:]].strip() if x else '',b.split(' ')))
    nameb = ' '.join(listb)


    if len(lista) <=len(listb):
        sim = sum(map(lambda x: x in nameb, namea.split(' '))) / len(namea.split(' '))
    else:
        sim = sum(map(lambda x: x in namea, nameb.split(' '))) / len(nameb.split(' '))
    print('sem sim: ', sim)
    return 1-sim

def jaccard(a, b):
    a = set(a)
    b = set(b)
    return len(a&b) / len(a|b)

def class_disimilarity(a, b):
    # print(',,,',a.split(' '))
    # print(',,,',b.split(' '))
    if not a or not b:
        # print(1)
        return 1
    classa = []
    for x in a.split(' '):
        classa.append(id_cat[x[1:]])
    # map(lambda x: classa.append(id_cat[x[1:]]), a.split(' '))
    classb = []
    for x in b.split(' '):
        classb.append(id_cat[x[1:]])
    # map(lambda x: classb.append(id_cat[x[1:]]), b.split(' '))
    print(classa)
    print(classb)
    if len(classa) <=len(classb):
        maxsim = 0
        for c in classa:
            sim = max(map(lambda x: jaccard(c, x), classb))
            if sim > maxsim:
                maxsim = sim
    else:
        maxsim = 0
        for c in classb:
            sim = max(map(lambda x: jaccard(c, x), classa))
            if sim > maxsim:
                maxsim = sim
    print('cat sim: ', maxsim)
    return 1-maxsim

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
            # if valuea
            # if not any(i in valuea.split(' ')  for i in valueb.split(' ')) or not valuea or not valueb:
            print('column a--------------->>>>', valuea)
            print('column b--------------->>>>', valueb)
            # print(semantic_disimilarity(valuea, valueb))
            # print(class_disimilarity(valuea, valueb))
            if 'd' in valuea or 'd' in valueb:
                print('include d')
                sum+= semantic_disimilarity(valuea, valueb)
            else:
                print('not include d')
                sum +=0.3 * semantic_disimilarity(valuea, valueb) + 0.7*class_disimilarity(valuea, valueb)

        dis[r] = sum/4
    print(dis)
    return dis

# global id_name
# global id_cat
# id_name, id_cat = init('n&c')
# reproduce results on small data set
# x = np.genfromtxt('data_names/dataset_extract.csv', dtype=str, delimiter=',')[:, 0:]
# y = np.genfromtxt('data_names/dataset_extract.csv', dtype=str, delimiter=',', usecols=(0 ))
x = np.genfromtxt('data_category/dataset_extract.csv', dtype=str, delimiter=',')[:, 0:]  # test.csv
y = np.genfromtxt('data_category/dataset_extract.csv', dtype=str, delimiter=',', usecols=(0 ))
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


np.savetxt('labels.out',kmodes_huang.labels_,  fmt='%i',delimiter=',')
np.savetxt('centroids.out',kmodes_huang.cluster_centroids_, fmt='%s',delimiter=',')

for result in (kmodes_huang,): #, kmodes_cao):
    # with open('cluster_results','w') as wf:
    classtable = np.zeros((dataNum,n_clusters), dtype=int)
    for ii, _ in enumerate(y):

        classtable[int(y[ii][-1]) - 1, result.labels_[ii]] += 1
    np.savetxt('classtable.out', classtable, fmt='%i', delimiter=',')
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
