import util
from sudoku import SudokuSearchProblem
from maps import MapSearchProblem

################ Node structure to use for the search algorithm ################
class Node:
    def __init__(self, state, action, path_cost, parent_node, depth):
        self.state = state
        self.action = action
        self.path_cost = path_cost
        self.parent_node = parent_node
        self.depth = depth

########################## DFS for Sudoku ########################
## Choose some node to expand from the frontier with Stack like implementation
def sudokuDepthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Return the final values dictionary, i.e. the values dictionary which is the goal state  
    """

    def convertStateToHash(values):
        """ 
        values as a dictionary is not hashable and hence cannot be used directly in the explored/visited set.
        This function changes values dict into a unique hashable string which can be used in the explored set.
        You may or may not use this
        """
        l = list(sorted(values.items()))
        modl = [a+b for (a, b) in l]
        return ''.join(modl)

    ## YOUR CODE HERE
    state = problem.getStartState()
    explored = set([convertStateToHash(state)])
    stack = []

    while not problem.isGoalState(state):
        stack = stack + list(map(lambda x: x[0], problem.getSuccessors(state)))
        while convertStateToHash(state) in explored:
            state = stack.pop()
        explored.add(convertStateToHash(state))

    return state
    # util.raiseNotDefined()

######################## A-Star and DFS for Map Problem ########################
## Choose some node to expand from the frontier with priority_queue like implementation

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def heuristic(state, problem):
    # It would take a while for Flat Earther's to get accustomed to this paradigm
    # but hang in there.

    """
        Takes the state and the problem as input and returns the heuristic for the state
        Returns a real number(Float)
    """
    end_state = problem.end_node
    return util.points2distance(\
        ((problem.G.node[state]['x'], 0, 0), (problem.G.node[state]['y'], 0, 0)),
        ((problem.G.node[end_state]['x'], 0, 0), (problem.G.node[end_state]['y'], 0, 0))
    )

def AStar_search(problem, heuristic=nullHeuristic):

    """
        Search the node that has the lowest combined cost and heuristic first.
        Return the route as a list of nodes(Int) iterated through starting from the first to the final.
    """

    state = problem.getStartState()
    depth = 0
    parentNode = Node(state, None, 0, None, depth)
    frontier = util.PriorityQueue()
    explored = set([state])

    while not problem.isGoalState(state):
        depth += 1

        for successor in problem.getSuccessors(state):
            node = Node(successor[0], successor[1], successor[2] + parentNode.path_cost, parentNode, depth)
            frontier.push(node, node.path_cost + heuristic(successor[0], problem))

        while parentNode.state in explored:
            parentNode = frontier.pop()

        explored.add(parentNode.state)
        state = parentNode.state

    route = []
    while parentNode is not None:
        route.append(parentNode.state)
        parentNode = parentNode.parent_node

    route.reverse()
    return route