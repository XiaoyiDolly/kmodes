
#!/usr/bin/env python

import numpy as np
from kmodes.kmodes import KModes


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


def semantic_disimilarity(a, b):
    # print(a)
    # print(b)
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
    # print('sem sim: ', sim)
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
    # print(classa)
    # print(classb)
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
    # print('cat sim: ', maxsim)
    return 1-maxsim

def multimatch_dissim(a, b, dec_map):
    nattrs = a.shape[1]
    stra = np.empty(a.shape).astype('object') # np.ones([a.shape[0], a.shape[1]], dtype=str)
    strb = np.empty(b.shape).astype('object') # np.ones([b.shape[0], ], dtype=str)

    for iattr in range(nattrs):
        att_str = dec_map[iattr]
        squarer = lambda t: att_str[t]
        vfunc = np.vectorize(squarer)
        strb[iattr] = vfunc(b[iattr])
        stra[:, iattr] = vfunc(a[:, iattr])
    dis = np.empty([a.shape[0],], dtype=int)

    for r,vr in enumerate(stra):
        sum = 0
        # print(vr)
        for c, vc in enumerate(vr):
            # print(vc)

            valuea = str(vc).strip()
            valueb = str(strb[c]).strip()
            if 'd' in valuea or 'd' in valueb:
                # print('include d')
                sum+= 0.4*semantic_disimilarity(valuea, valueb)
            elif 'i' in valuea or 'i' in valueb:
                # print('not include d')
                sum +=0.4*(class_disimilarity(valuea, valueb))#0.3 * semantic_disimilarity(valuea, valueb) + 0.8*
            else:
                # print('not include d')
                sum += 0.1 * (class_disimilarity(valuea, valueb))# 0.3 * semantic_disimilarity(valuea, valueb) + 0.7 *

        dis[r] = sum
    # print(dis)
    return dis

# x = np.genfromtxt('test.csv', dtype=str, delimiter=',')[:, 0:]  # test.csv
# y = np.genfromtxt('test.csv', dtype=str, delimiter=',', usecols=(0 ))
x = np.genfromtxt('data_category/dataset_extract.csv', dtype=str, delimiter=',')[:, 0:]  # test.csv
y = np.genfromtxt('data_category/dataset_extract.csv', dtype=str, delimiter=',', usecols=(0 ))#data_category/dataset_extract.csv

print(x.shape)
print(y.shape)
dataNum = y.shape[0]
n_clusters = 100
kmodes_huang = KModes(n_clusters=n_clusters, cat_dissim=multimatch_dissim, init='Huang', verbose=0)
kmodes_huang.fit(x)

# Print cluster centroids of the trained model.
print('k-modes (Huang) centroids:')
print(kmodes_huang.cluster_centroids_)
# Print training statistics
print('Final training cost: {}'.format(kmodes_huang.cost_))
print('Training iterations: {}'.format(kmodes_huang.n_iter_))


print('Save tables:')
np.savetxt('labels.out',kmodes_huang.labels_,  fmt='%i',delimiter=',')
np.savetxt('centroids.out',kmodes_huang.cluster_centroids_, fmt='%s',delimiter=',')

result = {}
for i, d in enumerate(y):
    clus = kmodes_huang.labels_[i]
    if clus not in result:
        result[clus] = []
    result[clus].append(d)

# print(result)
with open('clusters.txt','w') as f:
    for id, values in result.items():
        f.write(str(id)+':')
        f.write(" ".join(values))
        f.write('\n')
