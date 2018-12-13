import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def schwefel_func(x):
    x = np.array(x)
    return np.dot(-x, np.sin(np.sqrt(np.absolute(x))))

# print(schwefel_func([420.9687, 420.9687]))

# # Plotting 2D-SF
# x = np.arange(-500, 500, 0.5)
# y = np.arange(-500, 500, 0.5)
# xx, yy = np.meshgrid(x, y, sparse=False)
#
# # print(x)
# # print(y)
# # print(xx)
# # print(yy)
#
# z = np.zeros((len(x), len(y)))
# for i in range(len(x)):
#     for j in range(len(y)):
#         z[i, j] = schwefel_func([xx[i, j], yy[i, j]])
#
# # TODO check if z is correct with small number of vals
# # TODO check contour plot with plot in handout
#
# # print(z)
# print(z.shape)
# print(xx.shape)
# print(yy.shape)
#
# fig = plt.figure(1)
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(xx, yy, z, cmap=cm.coolwarm)
#
#
# # Plotting 2D-SF Contours
# fig2 = plt.figure(2)
# plt.contour(xx, yy, z, 10)
#
# plt.show()
#
#
#
