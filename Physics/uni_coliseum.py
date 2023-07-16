import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.index_tricks import AxisConcatenator
from scipy.optimize.optimize import fmin
import scipy.constants as cons

from functools import reduce


# Max/min values/indices

a = np.array([1,2,3,4,5, np.nan, 89])

max_v = np.max(a)
min_v = np.min(a)

max_ind = np.argmax(a)
min_ind = np.argmin(a)

# Ignores the NaN. However, better to remove first
nan_accounted_max = np.nanmax(a)

# Slicing, start at: end before, plus step size

smaller = a[0:5:2]

# first_row = a[0,:]

#-----------------------------------------------------------------------------

linear = np.arange(0, 30, step=3)

# Only works for 1D arrays

adjusted_ind = np.where((4 < linear) & (linear < 20))

adjusted = linear[adjusted_ind]

#-----------------------------------------------------------------------------

a = np.array([1,2,3,4,5,-3,3,-1])

b = np.where(a > 0, a, np.nan)

#-----------------------------------------------------------------------------

# data = np.genfromtxt('/home/marawan/angsty_algos/physics/assignments/z_boson_data_1.csv', delimiter=',', skip_header=1)
# data2 = np.genfromtxt('/home/marawan/angsty_algos/physics/assignments/z_boson_data_2.csv', delimiter=',', skip_header=1)

# first_filter = np.where((np.isfinite(data)) & (data >= 0))
# second_filter = np.where(np.isnan(data))
# third_filter = np.where(data < 0)

# NaN check
nan = np.isnan(linear)
# Not NaN check
not_nan = np.isfinite(linear)

#-----------------------------------------------------------------------------

j = np.array([1,2,3,4,5,5])
k = np.array([3,4,5,6,4])

big_one = np.hstack((j, k))

#-----------------------------------------------------------------------------

x_values = np.linspace(0, 10, 20)

def polynomial(x):
    return (x - 5)**2

x_max = 6

# plt.scatter(x_max, polynomial(x_max), label="start point")

# results = fmin(lambda x: polynomial(x), x_max, full_output=True)

# x_min = results[0][0]
# y_at_xmin = results[1]
# iterations = results[2]

#-----------------------------------------------------------------------------

# standard_deviation = np.std(data2[:, 1])
# mean = np.mean(data2[:, 1])

# stddev_filter = np.unique(np.where((data2[:, 1] - mean) > (3 * standard_deviation))[0])

#-----------------------------------------------------------------------------

b = np.array([
    [1,2,4,5,7,8,3,2],
    [6,43,6,78,4,3,7,6]
])

b = np.reshape(b, (8, 2))
c = np.sort(b, axis=0, kind='stable')

d = b[b[:, 0].argsort()]

#-----------------------------------------------------------------------------

# Floor division tests
num = 9
num2 = 21.4

#-----------------------------------------------------------------------------

num = 34.234556
r_num = round(num, 3)

#-----------------------------------------------------------------------------

def round_to_sig(num, sig_fig=3):
    one_sig_fig = - int(np.floor(np.log10(abs(num))))
    return round(num, one_sig_fig - 1 + sig_fig)

sig1 = round_to_sig(456.2345)
sig2 = round_to_sig(2000.23)


#-----------------------------------------------------------------------------

ev = cons.eV

#-----------------------------------------------------------------------------

def mesh_arrays(x_array, y_array):
    """
    """
    x_mesh = np.vstack([x_array]*y_array.shape[0])
    y_mesh = np.vstack([y_array]*x_array.shape[0]).transpose()

    return x_mesh, y_mesh

def quicker_mesh(x_array, y_array):
    return np.vstack([x_array]*y_array.shape[0]), np.vstack([y_array]*x_array.shape[0]).transpose()

x = np.array([1,2,3,4,5])
y = np.array([2,4,6,8,10,3,4,5])
z = np.empty((0, x.shape[0]))

#-----------------------------------------------------------------------------

blob = list(np.linspace(84, 96, 7))
blob2 = blob.append(91)


#-----------------------------------------------------------------------------

matrixx = [[3,6,2,1],
           [6,7,3,4],
           [5,3,7,8]]

# sorte = matrixx[np.argsort]

#------------------------------------------------------------------------------

grid = np.array([1,2,3,4,5])