import random
import math
import matplotlib.pyplot as plt
import matplotlib

#gets the dictionary {node:(Xcoord, Ycoord)} from a .tsp file
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

#Genetic program class with modifyable constants and all steps
class GP():
    def __init__(self, coord_dict):
        self.coords = coord_dict
        self.population = []

        #can modify these constants
        #make sure the number of elites and tournament size is <= the total size of the population!
        self.mutation_chance = 0.05
        self.population_size = 100
        self.tournament_size = 10
        self.max_iters = 1000
        self.elites = 30


    #The main method. Set graph to True if you also want to output a graph of the convergence
    def main_method(self, graph = False):
        x = []
        y = []
        i = 0

        #makes initial generation of random sequences
        self.first_generation(graph)

        #makes a new generation iters number of times
        for i in range(self.max_iters):
            self.make_new_generation()

            #setting up the graph values
            x.append(i)
            y.append(min(self.population, key = lambda x: x[1])[1])
            i+=1


        #prints graph if directed
        if graph:
            plt.plot(x, y,c="g")

            plt.xlabel("Iteration number")
            plt.ylabel("Current best value")

            plt.show()

        #returns the fittest individual in the final generation
        return min(self.population, key = lambda x: x[1])

    #makes a greedy guess of the best individual
    def make_initial_guess(self):
        path = []
        node_list = list(self.coords.keys())
        current_node = random.choice(node_list)
        node_list.remove(current_node)
        path.append(current_node)
        while len(node_list)>0:
            closest_node = min(node_list, key = lambda x: self.greedy(path[-1], x))
            path.append(closest_node)
            node_list.remove(closest_node)
        return path

    #the distance between two nodes
    def greedy(self, node1, node2):
        return ((self.coords[node2][1]-self.coords[node1][1])**2+(self.coords[node2][0]-self.coords[node1][0])**2)**.5

    #makes a generation with the greedy choice best individual and random choices
    def first_generation(self, graph):
        template = []
        if not graph:
            first = self.make_initial_guess()
            self.population.append((first, self.distance(first)))
        for element in self.coords.keys():
            template.append(element)
        for i in range(self.population_size-1):
            random_pop = random.sample(template, len(template))
            self.population.append((random_pop, self.distance(random_pop)))

    #The fitness function: the distance of the path (connecting back to the first node)
    def distance(self, path):
        total = 0
        cycle = path + [path[0]]
        for i in range(len(cycle)-1):
            distance = ((self.coords[cycle[i]][1]-self.coords[cycle[i+1]][1])**2 + (self.coords[cycle[i]][0]-self.coords[cycle[i+1]][0])**2)**.5
            total+=distance
        return total

    def make_new_generation(self):
        new_population = []

        #elitism, keep the top x fittest individuals
        new_population.extend(sorted(self.population, key = lambda x: x[1])[:self.elites])

        #keep adding children until population is full
        while len(new_population) < self.population_size:
            child = self.order_crossover(self.select_parents_tournament())

            #set chance to mutate
            if random.random() < self.mutation_chance:
                new_child = self.mutated_swap_elements(child)
                new_population.append((new_child, self.distance(new_child)))
            else:
                new_population.append((child, self.distance(child)))

        self.population = new_population

    #selects two parents from n-tournament select
    def select_parents_tournament(self):
        pool = []
        for i in range(self.tournament_size):
            pool.append(random.choice(self.population))
        parent1 = min(pool, key =lambda x: x[1])

        pool = []
        for i in range(self.tournament_size):
            pool.append(random.choice(self.population))
        parent2 = min(pool, key =lambda x: x[1])

        return (parent1[0], parent2[0])

    #returns new child using order crossover given two parents
    def order_crossover(self, parents):
        start_pos = random.randint(0,len(parents[0]))
        end_pos = random.randint(0,len(parents[1]))
        index_range = sorted([start_pos, end_pos])

        child = len(parents[0])*[None]
        seen = set([])
        for i in range(index_range[0], index_range[1]):
            child[i] = parents[0][i]
            seen.add(parents[0][i])        
        
        for i in range(len(parents[1])):
            if parents[1][i] not in seen:
                next_available_index = self.next_available_index(child)
                child[next_available_index] = parents[1][i]
        return child

    #helper function to find available indices in child
    def next_available_index(self, child):
        for i in range(len(child)):
            if child[i] is None:
                return i

    #mutation function: swaps two random nodes in the given path
    def mutated_swap_elements(self, child):
        index1 = random.randint(0, len(child)-1)
        index2 = random.randint(0, len(child)-1)
        placeholder = child[index1]

        child[index1] = child[index2]
        child[index2] = placeholder

        return child
        
#main method, put a file in here and it returns the best found path                      
def main(filepath, have_graph = False):
    graph= get_graph_from_file(filepath)
    gp = GP(graph)
    return gp.main_method(graph = have_graph)
                
#UNCOMMENT THESE TO RUN
#print(main("tiny.txt"))
#print(main("a280.tsp"))
#print(main("a280.tsp", have_graph = True))

                                   
