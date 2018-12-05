#Uses a minimum spanning tree to estimate a solution to the travelling salesman problem.

import sys
def distance(node1, node2):
    return ((node2[2]-node1[2])**2+(node2[1]-node1[1])**2)**.5

#A class of graph that is defined as a list of all vertices V
#and an adjacency matrix sized VxV.
class Graph():
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]  
                    for row in range(vertices)]

    #given node u and node v and the distance between them w, adds that edge.
    def add_edge(self, u, v, w):
        self.graph[u-1][v-1] = w


    # A utility function to return the constructed MST 
    def printMST(self, parent): 
        result = []
        for i in range(1,self.V):
            result.append([parent[i]+1, i+1, self.graph[i][parent[i]]])
        return result
  
    # A utility function to find the vertex with  
    # minimum distance value, from the set of vertices  
    # not yet included in shortest path tree 
    def minKey(self, key, mstSet): 
  
        # Initilaize min value 
        min_value = sys.maxsize 
  
        for v in range(self.V): 
            if key[v] < min_value and mstSet[v] == False: 
                min_value = key[v] 
                min_index = v 
  
        return min_index 
  
    # Function to construct and print MST for a graph  
    # represented using adjacency matrix representation 
    def primMST(self): 
  
        #Key values used to pick minimum weight edge in cut 
        key = [sys.maxsize] * self.V 
        parent = [None] * self.V # Array to store constructed MST 
        # Make key 0 so that this vertex is picked as first vertex 
        key[0] = 0 
        mstSet = [False] * self.V 
  
        parent[0] = -1 # First node is always the root of 
  
        for cout in range(self.V): 
  
            u = self.minKey(key, mstSet) 
  
            # Put the minimum distance vertex in  
            # the shortest path tree 
            mstSet[u] = True
   
            for v in range(self.V): 
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]: 
                        key[v] = self.graph[u][v] 
                        parent[v] = u 
  
        return self.printMST(parent)


#MAIN FUNCTION
def get_MST_heuristic(filepath):

    #parse the file into a Graph object
    f = open(filepath, "r")
    coord_list = []
    begun = False
    for line in f:
        if line.strip().startswith("1 "):
            begun = True
        if begun==True and line.strip() != "EOF":
            split = line.split()
            coord_list.append([int(split[0]), float(split[1]), float(split[2])])
    f.close()
    g = Graph(len(coord_list))
    for node1 in coord_list:
        for node2 in coord_list:
            dist = distance(node1, node2)
            g.add_edge(node1[0], node2[0], dist)
            g.add_edge(node2[0], node1[0], dist)

    #make the MST using Prim's
    MST = g.primMST()
    
    #double the path
    other_half = []
    for node in MST:
        other_half.append([node[1], node[0], node[2]])
    MST = MST + other_half

    MST_graph = Graph(len(coord_list))
    for node in MST:
        MST_graph.add_edge(node[0], node[1], node[2])
        MST_graph.add_edge(node[1], node[0], node[2])

    #traverse the MST using DFS, guarantees all nodes in Eulerian order
    long_path = DFS(MST, [MST[0][0], MST[0][0], 0])

    #remove nodes that have already been visited for final path
    seen  = []
    for node in long_path:
        if node not in seen:
            seen.append(node)
    seen.append(seen[0])

    #compute the final distance
    total = 0
    for i in range(len(seen)-1):
        total += g.graph[seen[i]-1][seen[i+1]-1]
    return seen, total

def DFSUtil(graph, v, visited, path, root): 

    # Mark the current node as visited
    visited.append(v[0])
    path.append(v[0])
    seen = False
    # Recur for all the vertices adjacent to this vertex 
    for u in graph:
        if u[0] == v[1] and u[1] not in visited:
            seen = True
            DFSUtil(graph, [u[1], u[1], 0], visited, path, root)

    return path


def DFS(graph, v):  

    # Call the recursive helper function for DFS
    return DFSUtil(graph, v, [], [], v[0]) 
    
