import numpy as np
import matplotlib.pyplot as plt
from utils import DLT
plt.style.use('seaborn')


pose_keypoints = np.array([16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28])

def read_keypoints(filename):
    fin = open(filename, 'r')

    kpts = []
    while(True):
        line = fin.readline()
        if line == '': break

        line = line.split()
        line = [float(s) for s in line]

        line = np.reshape(line, (len(pose_keypoints), -1))
        kpts.append(line)

    kpts = np.array(kpts)
    return kpts

def anglele(a,b,c):
    
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle =(np.degrees(angle))

    return angle


def visualize_3d(p3ds):

    """Now visualize in 3D"""
    torso = [[0, 1] , [1, 7], [7, 6], [6, 0]]
    armr = [[1, 3], [3, 5]]
    arml = [[0, 2], [2, 4]]
    legr = [[6, 8], [8, 10]]
    legl = [[7, 9], [9, 11]]
    body = [torso, arml, armr, legr, legl]
    colors = ['red', 'blue', 'green', 'black', 'orange']
    


    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for framenum, kpts3d in enumerate(p3ds):
        if framenum%2 == 0: continue #skip every 2nd frame
        for bodypart, part_color in zip(body, colors):
            for _c in bodypart:
                ax.plot(xs = [kpts3d[_c[0],0], kpts3d[_c[1],0]], ys = [kpts3d[_c[0],1], kpts3d[_c[1],1]], zs = [kpts3d[_c[0],2], kpts3d[_c[1],2]], linewidth = 4, c = part_color)
                #print(kpts3d[_c,:])
        #print(body)

            print("right leg:"+str(anglele(kpts3d[body[4][0][0]],kpts3d[body[4][0][1]],kpts3d[body[4][1][1]])))
            print("left leg:"+str(anglele(kpts3d[body[3][0][0]],kpts3d[body[3][0][1]],kpts3d[body[3][1][1]])))

            
            

        #uncomment these if you want scatter plot of keypoints and their indices.
        # for i in range(12):
        #     #ax.text(kpts3d[i,0], kpts3d[i,1], kpts3d[i,2], str(i))
        #     #ax.scatter(xs = kpts3d[i:i+1,0], ys = kpts3d[i:i+1,1], zs = kpts3d[i:i+1,2])


        #ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim3d(-100, 100)
        ax.set_xlabel('x')
        ax.set_ylim3d(-100, 100)
        ax.set_ylabel('y')
        ax.set_zlim3d(-200, 200)
        ax.set_zlabel('z')
        plt.pause(0.1)
        ax.cla()
    

if __name__ == '__main__':

    p3ds = read_keypoints('kpts_3d.dat')
    
    visualize_3d(p3ds)
  
