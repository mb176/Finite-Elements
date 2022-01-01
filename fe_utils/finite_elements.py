# Cause division to always mean floating point division.
from __future__ import division
import numpy as np
import copy
from .reference_elements import ReferenceInterval, ReferenceTriangle
np.seterr(invalid='ignore', divide='ignore')

def lagrange_points(cell, degree):
    """Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.

    The implementation of this function is left as an :ref:`exercise
    <ex-lagrange-points>`.

    """
    nodes = []
    # 1D case
    if cell.dim==1:  
        for idx in range(degree+1):
            nodes.append(cell.vertices[0]+idx/degree*(cell.vertices[1]-cell.vertices[0]))

    # 2D case, assumes a triangle spanned by 3 vertices
    elif cell.dim==2: 
        origin = cell.vertices[0]
        edge1 = cell.vertices[1]-cell.vertices[0]
        edge2 = cell.vertices[2]-cell.vertices[0]
        '''The triangle is spanned by edge1 and edge2 originating at origin. Their choice 
        matches the Referencetriangle.topology in such a way that LagrangePoints can 
        easily create entity_nodes in the right order.
        '''
        for i in range(degree+1): #index running along first triangle side
            for j in range(degree+1-i): #index running along second triangle side
                nodes.append(origin + j/degree*edge1 + i/degree*edge2)
    return np.array(nodes)
    

def vandermonde_matrix(cell, degree, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree on the cell provided.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct the matrix.
    :param points: a list of coordinate tuples corresponding to the points.
    :param grad: whether to evaluate the Vandermonde matrix or its gradient.

    :returns: the generalised :ref:`Vandermonde matrix <sec-vandermonde>` 
        of the form:
        grad = False: vandermonde[point, monomial]
        grad = True:  vandermonde[point, monomial, coordinate]   

    The implementation of this function is left as an :ref:`exercise
    <ex-vandermonde>`.
    """
    
    vandermonde = []
    #1D case
    if cell.name=='`ReferenceInterval`':
        x0 = np.array(points[:,0],dtype=float)

        #Loop over every column of the matrix
        for order in range(0,degree+1):
            #Append basis function/ gradient at the points
            if grad==False:
                vandermonde.append(x0**order)
            elif grad == True:
                vandermonde.append(np.nan_to_num(np.array([order*x0**(order-1)]).transpose()))
                
    #2D case
    elif cell.name=='ReferenceTriangle':
        x0 = np.array(points[:,0],dtype=float)
        y0 = np.array(points[:,1],dtype=float)

        #Loop over every column of the matrix
        for order in range(0,degree+1):
            for yPower in range(order+1):
                #Append basis function/ gradient at the points
                if grad==False:
                    vandermonde.append(x0**(order-yPower) * y0**yPower)
                elif grad == True:
                    xDerivative = np.nan_to_num((order-yPower)*x0**(order-yPower-1) * y0**yPower)
                    yDerivative = np.nan_to_num(x0**(order-yPower) * yPower *y0**(yPower-1))
                    vandermonde.append(np.array([xDerivative,yDerivative]).transpose())
                    

    else:
        raise Exception("Unknown cell type: %s"%cell.name)
    
    if grad==False:
        return np.array(vandermonde).transpose(1,0) 
    elif grad ==True:
        return np.array(vandermonde).transpose(1,0,2) 



class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """A finite element defined over cell.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the com
            ete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            the nodes of the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with entity `(d, i)`.

        Most of the implementation of this class is left as exercises.
        """

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = degree
        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = nodes
        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.
        self.entity_nodes = entity_nodes

        if entity_nodes:
            #: ``nodes_per_entity[d]`` is the number of entities
            #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array([len(entity_nodes[d][0])
                                              for d in range(cell.dim+1)])

        # A matrix cof oefficients for the polynomials spanning the nodal basis 
        # corresponding to nodes
        self.basis_coefs = np.linalg.inv(vandermonde_matrix(cell, degree, nodes))

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``."""

        #Calculate tabulation table by using the vandermonde matrix of points and 
        #multiplying by basis coefficients
        if grad == False: #Tabulate points
            table = np.dot(vandermonde_matrix(self.cell, self.degree, points),self.basis_coefs)
        elif grad == True: #Tabulate gradients
            table = np.einsum('ijk,jl->ilk', vandermonde_matrix(self.cell,self.degree,points,grad=True), 
                                self.basis_coefs)
        return table


    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        """
        return np.array(list(map(fn,self.nodes)))

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.

        The implementation of this class is left as an :ref:`exercise
        <ex-lagrange-element>`.
        """
        nodes = lagrange_points(cell, degree)
        # Use lagrange_points to obtain the set of nodes.  Once you
        # have obtained nodes, the following line will call the
        # __init__ method on the FiniteElement class to set up the
        # basis coefficients.

        '''Construct the entity_nodes dictionary
        :nodes_entity[d][n] returns all nodes contained in the entity denoted 
        by the tuple (d,n)
        '''
        entity_nodes = copy.deepcopy(cell.topology)
        #Start with the topology dictionary, then overwrite the vertices with nodes

        nodeUsed = np.zeros(len(nodes)) 
        #Flag array for the nodes, 0 if the node is associated to no entity yet

        #Loop over all entities:
        for dim in cell.topology:
            for entityIdx in cell.topology[dim]:
                #Test for each node if it is unused and part of the entity:
                entity_nodes[dim][entityIdx]=[]
                for nodeIdx in range(len(nodes)):
                    if (nodeUsed[nodeIdx]==False and 
                        cell.point_in_entity(nodes[nodeIdx],[dim,entityIdx])):

                        #If true, append the node to the entity_node list 
                        entity_nodes[dim][entityIdx].append(nodeIdx)
                        nodeUsed[nodeIdx]=True

                
        super(LagrangeElement, self).__init__(cell, degree, nodes, entity_nodes=entity_nodes)
