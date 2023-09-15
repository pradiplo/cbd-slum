import numpy as np
import math
import matplotlib.pyplot as plt
import itertools

class Node:
    children = None
    mass = None
    center_of_mass = None
    bbox = None
    vx = vy = None

def quad_insert(root, x, y, m):
    if root.mass is None:   #when the root is empty, add the first particle
        root.mass = m
        root.center_of_mass = [x,y]
        return
    elif root.children is None:
        root.children = [None,None,None,None]
        old_quadrant = quadrant_of_particle(root.bbox, root.center_of_mass[0], root.center_of_mass[1])
        if root.children[old_quadrant] is None:
            root.children[old_quadrant] = Node()
            root.children[old_quadrant].bbox = quadrant_bbox(root.bbox,old_quadrant)
        quad_insert(root.children[old_quadrant], root.center_of_mass[0], root.center_of_mass[1], root.mass)
        new_quadrant = quadrant_of_particle(root.bbox, x, y)
        if root.children[new_quadrant] is None:
            root.children[new_quadrant] = Node()
            root.children[new_quadrant].bbox = quadrant_bbox(root.bbox,new_quadrant)
        quad_insert(root.children[new_quadrant], x, y, m)
        root.center_of_mass[0] = (root.center_of_mass[0]*root.mass + x*m) / (root.mass + m)
        root.center_of_mass[1] = (root.center_of_mass[1]*root.mass + y*m) / (root.mass + m)
        root.mass = root.mass + m
    else:
        new_quadrant = quadrant_of_particle(root.bbox, x, y)
        if root.children[new_quadrant] is None:
            root.children[new_quadrant] = Node()
            root.children[new_quadrant].bbox = quadrant_bbox(root.bbox, new_quadrant)
        quad_insert(root.children[new_quadrant], x, y, m)
        root.center_of_mass[0] = (root.center_of_mass[0]*root.mass + x*m) / (root.mass + m)
        root.center_of_mass[1] = (root.center_of_mass[1]*root.mass + y*m) / (root.mass + m)
        root.mass = root.mass + m




def distance(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

def find_root_bbox(array):
    """ Create a suitable square boundary box for the input particles
    """
    if len(array) == 0 or len(array) == 1:
        return None
    xmin, xmax, ymin, ymax = array[0][1], array[0][1], array[0][2], array[0][2]
    for i in range(len(array)):
        if array[i][1] > xmax:
            xmax = array[i][1]
        if array[i][1] < xmin:
            xmin = array[i][1]
        if array[i][2] > ymax:
            ymax = array[i][2]
        if array[i][2] < ymin:
            ymin = array[i][2]
    if xmax - xmin == ymax - ymin:
        return xmin, xmax, ymin, ymax
    elif xmax - xmin > ymax - ymin:
        return xmin, xmax, ymin, ymax+(xmax-xmin-ymax+ymin)
    else:
        return xmin, xmax+(ymax-ymin-xmax+xmin), ymin, ymax

def quadrant_of_particle(bbox, x, y):
    """Return position of quadrant of the particle (x,y)
    """
    if y >= (bbox[3] + bbox[2])/2:
        if x <= (bbox[1] + bbox[0])/2:
            return 0
        else:
            return 1
    else:
        if x >= (bbox[1] + bbox[0])/2:
            return 2
        else:
            return 3

def quadrant_bbox(bbox,quadrant):
    """Return the coordinate of the quadrant
    """
    x = (bbox[0] + bbox[1]) / 2
    y = (bbox[2] + bbox[3]) / 2
    #Quadrant 0: (xmin, x, y, ymax)
    if quadrant == 0:
        return bbox[0], x, y, bbox[3]
    #Quadrant 1: (x, xmax, y, ymax)
    elif quadrant == 1:
        return x, bbox[1], y, bbox[3]
    #Quadrant 2: (x, xmax, ymin, y)
    elif quadrant == 2:
        return x, bbox[1], bbox[2], y
    #Quadrant 3: (xmin, x, ymin, y)
    elif quadrant == 3:
        return bbox[0], x, bbox[2], y

def get_distance(root,x,y,m,theta):
    if root.mass is None:
        pass
        #return 0
    if root.center_of_mass[0] == x and root.center_of_mass[1] == y and root.mass == m:
        #return [0]
        pass
        #return 0
    d = root.bbox[1]-root.bbox[0]
    r = distance(x,y, root.center_of_mass[0], root.center_of_mass[1])
    if d/r < theta or root.children is None:
        return[r]
    else:
        #print("children")
        dist = []
        #dist = 0
        c = 0
        for i in range(4):
            if root.children[i] is not None:
                dist.append(get_distance(root.children[i],x,y,m,theta))
                #dist += get_distance(root.children[i],x,y,m,theta)
                #if dist >0:
                c+=1
        #print(np.array(dist))       
        #return dist/c
        return dist
        #return np.mean(np.array(dist))

def compute_force(root,x,y,m,theta):
    if root.mass is None:
        return 0, 0
    if root.center_of_mass[0] == x and root.center_of_mass[1] == y and root.mass == m:
        return 0, 0
    d = root.bbox[1]-root.bbox[0]
    r = distance(x,y, root.center_of_mass[0], root.center_of_mass[1])
    if r/d > theta or root.children is None:
        return force(m, x, y, root.mass, root.center_of_mass[0], root.center_of_mass[1])
    else:
        fx = 0.0
        fy = 0.0
        for i in range(4):
            if root.children[i] is not None:
                fx += compute_force(root.children[i],x,y,m,theta)[0]
                fy += compute_force(root.children[i],x,y,m,theta)[1]
        return fx, fy

def avg_distance_between_bodies(bodies):
    total_distance = 0
    num_of_bodies = len(bodies)
    for i in range(num_of_bodies):
        for j in range(i+1, num_of_bodies):
            total_distance += distance(bodies[i][0], bodies[i][1], bodies[j][0], bodies[j][1])
    avg_distance = total_distance / (num_of_bodies * (num_of_bodies - 1) / 2)
    return avg_distance
    
def flatten(x):
    ''' Creates a generator object that loops through a nested list '''
    # First see if the list is iterable
    try:
        it_is = iter(x)
    # If it's not iterable return the list as is
    except TypeError:
        yield x
    # If it is iterable, loop through the list recursively
    else:
        for i in it_is:
            for j in flatten(i):
                yield j

def calc_avg_distance(particles,theta):
    n = len(particles)
    mass = np.ones(n)
    bodies = np.column_stack((mass,particles))
    #print(bodies)
    root = Node()
    root.center_of_mass = []
    root.bbox = find_root_bbox(bodies)
    tot_dist = []
    for i in range(n):
        quad_insert(root, bodies[i][1], bodies[i][2], bodies[i][0])
    for i in range(n):    
        dist = np.array(list(flatten(get_distance(root,bodies[i][1],bodies[i][2],bodies[i][0],theta))))
        #dist =get_distance(root,bodies[i][1],bodies[i][2],bodies[i][0],theta)
        #print(dist)
        for d in dist:
            tot_dist.append(d) 
        #tot_dist.append(dist)   
    #print(tot_dist) 
    return sum(tot_dist)/ len(tot_dist)

def calc_force_on_body(particles,x,y,m,theta):
    n = len(particles)
    mass = np.ones(n)
    bodies = np.column_stack((mass,particles))
    #print(bodies)
    root = Node()
    root.center_of_mass = []
    root.bbox = find_root_bbox(bodies)
    for i in range(n):
        quad_insert(root, bodies[i][1], bodies[i][2], bodies[i][0])
    total_fx, total_fy = compute_force(root,x,y,m,theta)
    return total_fx, total_fy


def brute_force(bodies, x, y, m):
    total_force = 0
    fx = 0
    fy = 0
    for body in bodies:
        if body[0] != x or body[1] != y:
            fx_, fy_ = force(m, x, y, 1, body[0], body[1])
            #total_force += math.sqrt(fx**2 + fy**2)
            fx += fx_
            fy += fy_
    return fx,fy

def force(m, x, y, mcm, xcm, ycm):
    d = distance(x, y, xcm, ycm)
    f = m*mcm/(d**2)
    dx = xcm - x
    dy = ycm - y
    angle = math.atan2(dy, dx)
    fx = math.cos(angle) * f
    fy = math.sin(angle) * f
    return fx, fy



if __name__ == '__main__':
    num_points = 500
    particles = np.random.rand(num_points,2)
    #mass = np.ones(num_points)
    #particles = np.column_stack((mass,points))
    # Scatter plot of particles position
    plt.scatter(particles[:, 0], particles[:, 1])
    plt.title('Particles Position')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    import time
    start_time = time.time()
    brute = avg_distance_between_bodies(particles)
    brute_fx,brute_fy = brute_force(particles,particles[0,0],particles[0,1],1)
    end_time = time.time()
    brute_time = end_time - start_time
    start_time = time.time()
    bh = calc_avg_distance(particles,0.0)
    bh_force_x,bh_force_y =calc_force_on_body(particles,particles[0,0],particles[0,1],1,0.0)
    end_time = time.time()
    bh_time = end_time - start_time
    print("Brute, time=" + str(brute_time) + " seconds, dist=" + str(brute))
    print("Barnes-hut, time=" + str(bh_time) +  " seconds, dist=" + str(bh))
