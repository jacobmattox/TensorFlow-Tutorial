
from random import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

m = randint(-10, 10)
n = randint(-10, 10)
b = randint(-10, 10)


all_x1 = [el for el in range(100)] * 100
shuffle(all_x1)

all_x2 = [el for el in range(100)] * 100
shuffle(all_x2)

all_y = [m * x1 + n * all_x2[idx] + b for idx, x1 in enumerate(all_x1)]



# plt.scatter(all_x1, all_x2)
# plt.show()

myFigure = plt.figure()

ax = myFigure.add_subplot(111, projection='3d')
ax.scatter(all_x1, all_x2, all_y)


#
# for el in range(100):
#     ax.view_init(10, el)
#     plt.savefig("rotate_" + str(el))

