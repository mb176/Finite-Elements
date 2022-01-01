from fe_utils.quadrature import *
from fe_utils.reference_elements import *
from fe_utils.finite_elements import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

def fn(x):
    return x**2

cell = ReferenceInterval
degree =3

vandermonde = vandermonde_matrix(cell, degree, lagrange_points(cell,degree),grad=True)
print(vandermonde.shape)
print(round(comb(degree + cell.dim, cell.dim)))
a = np.array([1,2,3])
b = np.array([a])
print(b.transpose())
a=np.ones(100000)

fe = LagrangeElement(cell, degree)

# Only test edges.
d = 1

for e, nodes in fe.entity_nodes[d].items():
    vertices = [np.array(cell.vertices[v]) for v in cell.topology[d][e]]

    # Project the nodes onto the edge.
    p = [np.dot(fe.nodes[n] - vertices[0], vertices[1] - vertices[0])
            for n in nodes]

    assert np.all(p[:-1] < p[1:])
