from fe_utils.quadrature import *
from fe_utils.reference_elements import *
from fe_utils.finite_elements import *
import numpy as np
import matplotlib.pyplot as plt

cell = ReferenceTriangle
#triangle = ReferenceTriangle
LagrangeElement(cell,1)

plt.show()