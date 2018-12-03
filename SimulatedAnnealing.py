import random
import math
import matplotlib.pyplot as plt
import matplotlib


#parses a file and puts the coords in a {node: (xval, yval)} dictionary
def get_graph_from_file(filepath):
    f = open(filepath, "r")
    coord_dict = {}
    begun = False
    for line in f:
        if line.strip().startswith("1 "):
            begun = True
        if begun==True and line.strip() != "EOF":
            split = line.split()
            coord_dict[int(split[0])] = [float(split[1]), float(split[2])]
    f.close()
    return coord_dict

#greedy choice to make a good initial solution to start from
def make_initial_guess(coord_dict):
    path = []
    node_list = list(coord_dict.keys())
    current_node = random.choice(node_list)
    node_list.remove(current_node)
    path.append(current_node)
    while len(node_list)>0:
        closest_node = min(node_list, key = lambda x: distance(path[-1], x, coord_dict))
        path.append(closest_node)
        node_list.remove(closest_node)
    return path

#random choice to make a random initial solution
def make_random_guess(coord_dict):
    path = list(coord_dict.keys())
    random.shuffle(path)
    return path

#returns a neighbor of path by switching two random nodes in path
def get_random_neighbor(path):
    neighbors = []
    for i in range(len(path)):
        for j in range(len(path)):
            if i<j:
                neighbors.append([path[0:i] + [path[j]] + path[i+1:j] + [path[i]] +path[j+1:]])

    return random.choice(neighbors)[0]

#euclidean distane between two nodes
def distance(node1, node2, coord_dict):
    return ((coord_dict[node2][1]-coord_dict[node1][1])**2+(coord_dict[node2][0]-coord_dict[node1][0])**2)**.5

#The C(s) function for simulated annealing, the fitness
#returns the distance a path takes
def C(path, coord_dict):
    total = 0
    cycle = path + [path[0]]
    for i in range(len(cycle)-1):
        distance = ((coord_dict[cycle[i]][1]-coord_dict[cycle[i+1]][1])**2 + (coord_dict[cycle[i]][0]-coord_dict[cycle[i+1]][0])**2)**.5
        total+=distance

    return total

#cooling function is multiplicative exponential where alpha = .9 for little loss       
def T(t, t_0):
    alpha = .9
    return t_0*(alpha**t)

#MAIN FUNCtION
def simulated_annealing(filepath, maxiters=1000):
    graph = get_graph_from_file(filepath)
    S_0 = make_initial_guess(graph)
    S = S_0

    #defining constants
    t_0 = 1
    k=1
    
    i = 0
    while i<maxiters:
        neighbor = get_random_neighbor(S)
        if C(neighbor, graph) <= C(S, graph):
            S = neighbor

        else:
            threshold = math.e**(-(C(neighbor, graph)-C(S, graph))/(k*T(i,t_0)))
            if random.random() < threshold:
                S = neighbor
        i+=1
    return S, C(S, graph)

        
#Same as simulated_annealing except it uses the random choice to start
#so the improvement is more clear to see and returns a graph as well
def simulated_annealing_graph(filepath, maxiters=1000):
    graph = get_graph_from_file(filepath)
    S_0 = make_random_guess(graph)
    S = S_0
    t_0 = 1
    k=1
    i = 0
    x = []
    y= []
    while i<maxiters:
        x.append(i)
        neighbor = get_random_neighbor(S)
        if C(neighbor, graph) <= C(S, graph):
            S = neighbor

        else:
            threshold = math.e**(-(C(neighbor, graph)-C(S, graph))/(k*T(i,t_0)))
            if random.random() < threshold:
                S = neighbor
        y.append(C(S,graph))
        i+=1

    plt.plot(x, y,c="g")

    plt.xlabel("Iteration number")
    plt.ylabel("Current best value")

    plt.show()
    return S, C(S, graph)

#UNCOMMENT THESE TO TEST

#print(simulated_annealing("tiny.txt"))   
#print(simulated_annealing("a280.tsp"))
#print(simulated_annealing_graph("a280.tsp"))



    
