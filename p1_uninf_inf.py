import heapq

class Vertex:
    """
    A class used to represent an Vertex of a Graph.

    ...

    Attributes
    ----------
    id : int
        ID of the Vertex
    adjacent : dictionary
        Represents set of neighboring vertices: adjacent vertice ID as
        dictionary key and weight of the edge as dictionary value.
    cell_id : int
        Represents the Square ID in which vertex is located.
    parent : int
        Parent of the given vertex.

    Methods
    -------
    add_neighbor(self, neighbor, weight=0)
        Adds neighbor and weight as (key,value) set to adjacent dictionary.
    add_cell_id(self, cell_id)
        Defines cell_id for the given vertex.
    get_neighbors(self)
        Gets list of adjacent vertices.
    get_id(self)
        Gets ID of the vertex.
    get_weight(self, neighbor)
        Gets weight of the (self, neighbor) edge.
    """

    
    def __init__(self, node):

        """
        Parameters
        ----------
        node : int
            The ID of the vertex (which node it is)

        """
        
        self.id = node
        self.adjacent = {}
        self.cell_id = 0
        self.parent = 0

    def add_neighbor(self, neighbor, weight=0):
        """Adds neighbor vertex to the adjacent dictionary with the
           given weight.

        If the argument `weight` isn't passed in, the default value is 0.

        Parameters
        ----------
        neighbor : int
            ID of the adjacent vertex.
        weight: int
            weight of the (self, neighbor) edge.
        
        """
        
        self.adjacent[neighbor] = weight

    def add_cell_id(self, cell_id):
        """Adds cell_id (square  id) for the given vertex.

        Parameters
        ----------
        cell_id : int
            ID of the square cell in which vertex is located.        
        """        
        self.cell_id = cell_id
    
    def get_neighbors(self):
        """Gets list of adjacent vertices."""

        return self.adjacent.keys()  

    def get_id(self):
        """ Gets ID of the vertex. """
        return self.id

    def get_weight(self, neighbor):
        """Gets weight of the (self, neighbor) edge.

        Parameters
        ----------
        neighbor : int
            ID of the adjacent vertex.        
        """          
        return self.adjacent[neighbor]

class Graph:
    """
    A class used to represent a Graph.

    ...

    Attributes
    ----------
    vertex_dict : dictionary
        ID of the Vertex

    Methods
    -------
    add_vertex(self, node, cell_id)
        Adds Vertex to the Graph with the given node(vertex ID) and
        cell_id (square ID).
    get_vertex(self, node)
        Returns vertex with the given ID.
    add_edge(self, frm, to, cost = 0)
        Adds edge (frm, to) with the given weight to the Graph.
    
    """

    
    def __init__(self):
        self.vertex_dict = {}

    def add_vertex(self, node, cell_id):
        """Adds Vertex to the Graph with the given node(vertex ID) and
           cell_id (square ID).

        Parameters
        ----------
        node : int
            Vertex ID.
        cell_id : int
            ID of the square cell in which vertex is located.
        """
        
        new_vertex = Vertex(node)
        self.vertex_dict[node] = new_vertex
        self.vertex_dict[node].add_cell_id(cell_id)
        return new_vertex

    def get_vertex(self, node):
        """Returns vertex with the given ID.

        Parameters
        ----------
        node : int
            Vertex ID        
        """
        
        if node in self.vertex_dict:
            return self.vertex_dict[node]
        else:
            return None

    def add_edge(self, frm, to, weight = 0):
        """Adds edge (frm, to) with the given weight to the Graph..

        Parameters
        ----------
        frm : int
            ID of the source node.
        to : int
            ID of the destination node.
        weight : int
            Weight of the (frm, to) edge
        """
        
        if frm not in self.vertex_dict:
            self.add_vertex(frm, weight)
        if to not in self.vertex_dict:
            self.add_vertex(to, weight)

        self.vertex_dict[frm].add_neighbor(self.vertex_dict[to], weight)
        self.vertex_dict[to].add_neighbor(self.vertex_dict[frm], weight)

def heuristic(frm, to):
    """Heuristic function.

    Calculates Manhattan distance between two nodes (frm, to).

    Parameters
    ----------
    frm : int
        ID of the source node.
    to : int
        ID of the destination node.

    Returns: Manhattan distance
    """
    
    from_cell = frm.cell_id
    to_cell = to.cell_id

    # from  cell IDs extract (1st digit + 1) as x coordinate
    # and (2nd digit + 1) as y coordinate
    x_frm = ((from_cell // 10) + 1) * 100
    y_frm = ((from_cell % 10) + 1) * 100

    x_to = ((to_cell // 10) + 1) * 100
    y_to = ((to_cell % 10) + 1) * 100

    dist = abs(x_to - x_frm) + abs(y_to - y_frm)
    
    return dist

def ucs(graph, start, end):
    """UCS algoritm.

    Parameters
    ----------
    graph : Graph()
        Graph in which shortest distance between two nodes should be found
    frm : int
        ID of the source node.
    to : int
        ID of the destination node.

    Returns: minimal cost for the shortest path
    """

    # get vertices with the given IDs from given graph
    start = graph.get_vertex(start)
    goal = graph.get_vertex(end)

    global parent
    parent = {}
    global visited
    visited = []
    global frontier
    frontier = []
	
    current_cost = {}
    parent[start.id] = None
    current_cost[start.id] = 0

    # first node in the queue is source node
    # current_cost[start.id] = g
    heapq.heappush(frontier,(current_cost[start.id], start.id))
    
    while frontier:
        cost, node_id = heapq.heappop(frontier)
        
        if node_id == goal.id:
            return current_cost[goal.id]
        
        node = graph.get_vertex(node_id)
            
        for ngbr in node.get_neighbors():
            new_cost = current_cost[node.id] + node.get_weight(ngbr)
            if ngbr.id not in current_cost or new_cost < current_cost[ngbr.id]:
                current_cost[ngbr.id] = new_cost

                # only value of g is considered in the priority queue
                # here g = new_cost
                heapq.heappush(frontier,(new_cost, ngbr.id))
                parent[ngbr.id] = node.id
                

def aStar(graph, start, end):
    """A* algoritm.

    Parameters
    ----------
    graph : Graph()
        Graph in which shortest distance between two nodes should be found
    frm : int
        ID of the source node.
    to : int
        ID of the destination node.

    Returns: minimal cost for the shortest path
    """

    # get vertices with the given IDs from given graph
    start = graph.get_vertex(start)
    goal = graph.get_vertex(end)

    global parent
    parent = {}
    current_cost = {}
    parent[start.id] = None
    current_cost[start.id] = 0
    
    global visited
    visited = []
    global frontier
    frontier = []
    f = 0

    # first node in the queue is source node
    heapq.heappush(frontier,(f, start.id))
    
    while frontier:
        cost, node_id = heapq.heappop(frontier)
        
        if node_id == goal.id:
            return current_cost[goal.id]
          
        node = graph.get_vertex(node_id)
            
        for ngbr in node.get_neighbors():
            new_cost = current_cost[node.id] + node.get_weight(ngbr)
            if ngbr.id not in current_cost or new_cost < current_cost[ngbr.id]:
                current_cost[ngbr.id] = new_cost
                f = new_cost + heuristic(node, ngbr)

                # value of the f = g + h considered in the priority queue
                # here g = new_cost, h = heuristic(node, ngbr)
                heapq.heappush(frontier,(f, ngbr.id))
                parent[ngbr.id] = node.id

        
def file_reader(filepath, graph):
    """Reads .txt file from the given path and constructs Graph

    Parameters
    ----------
    filepath : str
        Path of the input file
    graph : Graph()
        Graph which is should be constructed with the data extracted
        from given filepath
    """
    
    global source
    global dest

    with open(filepath, "r") as filestream:
        for line in filestream:

            if line.startswith("#"):
                continue
            else:
                line = line.rstrip('\n')
                cl = line.split(",")
                
                if len(cl)==2:
                    if cl[0] == 'S':
                        source = int(cl[1])
                    elif cl[0] == 'D':
                        dest = int(cl[1])
                    else:
                        g.add_vertex(int(cl[0]), int(cl[1]))
                elif len(cl) == 3:
                    g.add_edge(int(cl[0]), int(cl[1]), int(cl[2]))

def find_path(source, dest):
    """Finds the shortest path with the optimal cost for the given nodes.

    Parameters
    ----------
    source : int
        ID of the source node.
    dest : int
        ID of the destination node.

    Returns: The shortest path
    """    

    path = [str(dest)]
    p = parent[dest]
    
    while(p != source):
        path.append(str(p))
        temp = p
        p = parent[temp]
        
    path.append(str(source))
    path.reverse()
    return (' -> '.join(path))
    

if __name__ == '__main__':

    g = Graph()    
    inputpath = input('Enter your file path:  ')
    
    path = []
    source = 0
    dest = 0

    # construcnting g graph from input file
    file_reader(inputpath, g)

    choise = input('Start and end points from input file are ({}, {})\
                   .\nDo yo want to change them? (y/n) '\
                   .format(source, dest))

    if choise[0].lower() == 'y':
        source = int(input('Source: '))
        dest = int(input('Destination: '))

    # for source == dest case do not do additional calculations
    if source == dest:
        cost_ucs = cost_astar = 0
        shortest_path = str(source)
    else:
        cost_ucs = ucs(g, source, dest)
        cost_astar = aStar(g, source, dest)

        shortest_path = find_path(source, dest)        
    
    
    print('Optimal UCS cost from {} to {} is: {}'.format(source, dest, cost_ucs))    
    print('Optimal A* cost from {} to {} is: {}'.format(source, dest, cost_astar))    
    print('Shortest path is:', shortest_path)



    
