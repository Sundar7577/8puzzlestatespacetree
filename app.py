


from flask import Flask, render_template, request
from collections import deque

app = Flask(__name__)


class PuzzleState:
    def __init__(self, state,depth, parent=None, move="",fvalue=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.depth = depth
        self.fvalue = fvalue
        

    def __eq__(self, other):
        return self.state == other.state
    
    def __hash__(self) -> int:
        return hash(str(self.state))
    
    
    
##############################################################################
# Code For A* Search using Manhattan Distance
##############################################################################
    
    
def calculate_manhattan_distance(state:PuzzleState,goal_state:PuzzleState):
    size = 3  # Assuming a 3x3 puzzle
    # Convert the state list into a 2D list for easier indexing
    state_2d = [state.state[i:i+size] for i in range(0, size*size, size)]
    total_distance = 0

    for i in range(size):
        for j in range(size):
            if state_2d[i][j] != 0:
                # Calculate the Manhattan distance for each tile
                goal_position = divmod(goal_state.state.index(state_2d[i][j]), size)
                total_distance += abs(i - goal_position[0]) + abs(j - goal_position[1])

    return total_distance


def a_star(initial_state:PuzzleState, goal_state:PuzzleState):
        visited: set[PuzzleState] = set([initial_state])
        open:list = []
        closed:list = []
        initial_state.fvalue = calculate_manhattan_distance(state=initial_state, goal_state=goal_state)
        open.append(initial_state)
        while True:
            current_state = open[0]
            # for i in cur.data:
            #     for j in i:
            #         print(j, end=" ")
            #     print("")
            # if the difference between current and goal node is 0 we have reached the goal node
            if (calculate_manhattan_distance(current_state, goal_state) == 0):
                return visited
            for i in cleaned_generate_successors(current_state):
                i.fval = calculate_manhattan_distance(i, goal_state)
                open.append(i)
                visited.add(i)
            closed.append(current_state)
            del open[0]
            open.sort(key=lambda x: x.fval, reverse=False)



#######################################################################
# Children Generator Function
#######################################################################
def generate_successors(state):
    successors = []
    zero_index = state.state.index(0)
    moves = [-1, 1, -3, 3]  # Left, Right, Up, Down

    for move in moves:
        new_index = zero_index + move

        if (
            0 <= new_index < len(state.state)
            and not (
                (zero_index % 3 == 0 and move == -1)
                or (zero_index % 3 == 2 and move == 1)
            )
        ):
            new_state = state.state[:]
            new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
            successors.append(PuzzleState(new_state,1,  state, move))

    return successors


def cleaned_generate_successors(state:PuzzleState) -> list[PuzzleState]:
    successors:list[PuzzleState] = []
    zero_index = state.state.index(0)
    moves = [-1, 1, -3, 3]

    for move in moves:
        new_index = zero_index + move

        if (
            0 <= new_index < len(state.state)
            and not (
                (zero_index % 3 == 0 and move == -1)
                or (zero_index % 3 == 2 and move == 1)
            )
        ):
            new_state = state.state[:]
            new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
            successors.append(PuzzleState(new_state,state.depth+1,  state, move))

    return successors

#####################################################################################
# Breadth First Search Function
#####################################################################################
def cleaned_bfs(initial_state:PuzzleState, goal_state:PuzzleState):
    queue: deque[PuzzleState] = deque([initial_state])
    visited: set[PuzzleState] = set([initial_state])

    while queue:
        current_state:PuzzleState = queue.popleft()
        visited.add(current_state)

        if current_state == goal_state:
            return visited

        successors:list[PuzzleState] = cleaned_generate_successors(current_state)
        for successor in successors:
            if successor not in visited:
                queue.append(successor)
            
            

def bfs(initial_state, goal_state):
    queue = deque([(initial_state, 0)])
    visited = set([tuple(initial_state.state)])
    all_states = {}

    while queue:
        current_state, depth = queue.popleft()

        if depth not in all_states:
            all_states[depth] = []

        all_states[depth].append(current_state)

        if current_state.state == goal_state.state:
            return all_states

        successors = generate_successors(current_state)
        for successor in successors:
            if tuple(successor.state) not in visited:
                visited.add(tuple(successor.state))
                queue.append((successor, depth + 1))


#####################################################################################
# Depth First Search Function
#####################################################################################
def cleaned_dfs(initial_state, goal_state):
    stack: deque[PuzzleState] = deque([initial_state])
    visited: set[PuzzleState] = set([initial_state])

    while stack:
        current_state = stack.pop()
        visited.add(current_state)

        if current_state.state == goal_state.state:
            return list(visited)

        successors = cleaned_generate_successors(current_state)
        if current_state.depth < 8:  # Limit the depth
            for successor in successors:
                if successor not in visited:
                    stack.append((successor))

def dfs(initial_state, goal_state):
    stack = [(initial_state, 0)]
    visited = set([tuple(initial_state.state)])
    all_states = {}

    while stack:
        current_state, depth = stack.pop()

        if depth not in all_states:
            all_states[depth] = []

        all_states[depth].append(current_state)

        if current_state.state == goal_state.state:
            return all_states

        successors = generate_successors(current_state)
        if depth < 8:  # Limit the depth
            for successor in successors:
                if tuple(successor.state) not in visited:
                    visited.add(tuple(successor.state))
                    stack.append((successor, depth + 1))



######################################################################################
# Flask Specific:
# Renders web UI
######################################################################################

@app.route('/', methods=['GET', 'POST'])
def render_state_space_tree():
    if request.method == 'GET':
        initial = [2, 8, 3, 1, 6, 4, 7, 0, 5]
        goal = [2, 0, 8, 1, 6, 3, 7, 5, 4]
        initial_state = PuzzleState(initial,0)
        goal_state = PuzzleState(goal,0)

        explored_states: set[PuzzleState] = dfs(initial_state, goal_state)

        # Prepare the data for vis.js network
        nodes = []
        edges = []

        for depth, states_at_depth in explored_states.items():
            for state in states_at_depth:
                #add nodes
                nodes.append({"id": str(state.state), 'label': str(
                    state.state), "level": f"{depth}"})

                # Add edges (parent-child relationship)
                if state.parent:
                    edges.append({"from": str(state.parent.state),
                                 "to": str(state.state), 'arrows': 'to'})

        context = {"nodes": nodes, "edges": edges}
        return render_template("index.html", **context)

    if request.method == "POST":
        search_algorithm = request.form.get('search_algorithm')

        initial = [int(x) for x in request.form.get('initial').split(' ')]
        goal = [int(x) for x in request.form.get('final').split(' ')]
        initial_state = PuzzleState(initial,0)
        goal_state = PuzzleState(goal,0)

        if search_algorithm == 'bfs':
            explored_states = bfs(initial_state, goal_state)
        elif search_algorithm == 'dfs':
            explored_states = dfs(initial_state, goal_state)
        else:
            return "Invalid search algorithm"

        # Prepare the data for vis.js network
        nodes = []
        edges = []

        for depth, states_at_depth in explored_states.items():
            for state in states_at_depth:
                # Add nodes
                nodes.append({"id": str(state.state), 'label': str(
                    state.state), "level": f"{depth}"})

                # Add edges (parent-child relationship)
                if state.parent:
                    edges.append({"from": str(state.parent.state),
                                 "to": str(state.state), 'arrows': 'to'})

        context = {"nodes": nodes, "edges": edges,"algorithm":search_algorithm,"initial":initial,"goal":goal,}
        return render_template("index.html", **context)


@app.route('/cleaned', methods=['GET','POST'])
def render_cleaned_state_space_tree():
    if request.method == 'GET':
        initial = [2, 8, 3, 1, 6, 4, 7, 0, 5]
        goal = [2, 0, 8, 1, 6, 3, 7, 5, 4]
        initial_state = PuzzleState(initial,0)
        goal_state = PuzzleState(goal,0)

        explored_states: list[PuzzleState] = cleaned_bfs(initial_state, goal_state)
        

        # Prepare the data for vis.js network
        nodes = []
        edges = []

        for state in explored_states:
            nodes.append({"id": str(state.state), 'label': str(
                state.state), "level": f"{state.depth}"})
                # Add edges (parent-child relationship)
            if state.parent:
                edges.append({"from": str(state.parent.state),
                                "to": str(state.state), 'arrows': 'to'})

        context = {"nodes": nodes, "edges": edges,"algorithm":"bfs","initial":' '.join([str(x) for x in initial]),"goal":' '.join([str(x) for x in goal])}
        return render_template("index.html", **context)
    
    if request.method == "POST":
        search_algorithm = request.form.get('search_algorithm')

        initial = [int(x) for x in request.form.get('initial').split(' ')]
        goal = [int(x) for x in request.form.get('final').split(' ')]
        initial_state = PuzzleState(initial,0)
        goal_state = PuzzleState(goal,0)

        if search_algorithm == 'bfs':
            explored_states = cleaned_bfs(initial_state, goal_state)
        elif search_algorithm == 'dfs':
            explored_states = cleaned_dfs(initial_state, goal_state)
        elif search_algorithm == "astar":
            explored_states = a_star(initial_state,goal_state)
        else:
            return "Invalid search algorithm"

        # Prepare the data for vis.js network
        nodes = []
        edges = []

        for state in explored_states:
            nodes.append({"id": str(state.state), 'label': str(
                state.state), "level": f"{state.depth}"})
                # Add edges (parent-child relationship)
            if state.parent:
                edges.append({"from": str(state.parent.state),
                                "to": str(state.state), 'arrows': 'to'})

        context = {"nodes": nodes, "edges": edges,"algorithm":search_algorithm,"initial":' '.join([str(x) for x in initial]),"goal":' '.join([str(x) for x in goal]),}
        return render_template("index.html", **context)



if __name__ == "__main__":
    app.run(debug=True)
