import numpy
from plyfile import PlyData, PlyElement

plydata = PlyData.read('/home/mahedi/Desktop/3DGS/output/mipnerf360/bicycle/point_cloud/iteration_30000/point_cloud.ply')

x_data = plydata.elements[0].data['x'][0:10]
y_data = plydata.elements[0].data['y'][0:10]
z_data = plydata.elements[0].data['z'][0:10]

# print("distance between first two points", (x_data[0] - x_data[1]))
# print("distance between first two points", (x_data[0] - x_data[5]))
# print("distance between first two points", (x_data[0] - x_data[5]))

# for i in range (5):
#     print("distance between first two points", (x_data[0] - x_data[i]))

for i in range (10):
    print("distance between first two points", numpy.sqrt((x_data[0] - x_data[i])**2 + (y_data[0] - y_data[i])**2  + (z_data[0] - z_data[i])**2))


print(x_data, y_data)



scale_0 = plydata.elements[0].data['scale_0'][0]
scale_1 = plydata.elements[0].data['scale_1'][0]
scale_2 = plydata.elements[0].data['scale_2'][0]

print(scale_0, scale_1, scale_2)
