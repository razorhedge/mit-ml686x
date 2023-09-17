import numpy as np
import kmeans
import common
import naive_em
import em
from matplotlib import pyplot as plt

X = np.loadtxt("toy_data.txt")

# TODO: Your code here

K = [1,2,3,4]
seed = [0,1,2,3,4]
min_cost = 100000
for k in K:
    for s in seed:
        mixture, post = common.init(X, k, s)
        kmix, kpost, kcost = kmeans.run(X, mixture, post)
        common.plot(X, kmix, kpost, title='K={},seed={}'.format(k,s))
        print("Cost = {}".format(kcost))
        if kcost < min_cost:
            min_cost = kcost
    print("Min cost for K={}, {}".format(k, kcost))


for k in K:
    mixture, post = common.init(X, k)
    kmix, kpost, l_new = naive_em.run(X, mixture, post)
    common.plot(X, kmix, kpost, title='K={},seed={}'.format(k,s))
    print("Cost = {}".format(l_new))
