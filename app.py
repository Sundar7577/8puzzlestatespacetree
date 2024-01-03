


from flask import Flask, render_template, request
from collections import deque

app = Flask(__name__)


class PuzzleState:
    def __init__(self, state,depth, parent=None, move=""):
        self.state = state
        self.parent = parent
        self.move = move
        self.depth = depth
        

    def __eq__(self, other):
        return self.state == other.state
    
    def __hash__(self) -> int:
        return hash(str(self.state))


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
            successors.append(PuzzleState(new_state,state.depth+1,  state, move))

    return successors

def cleaned_bfs(initial_state:PuzzleState, goal_state:PuzzleState):
    queue: deque[PuzzleState] = deque([initial_state])
    visited: set[PuzzleState] = set([initial_state])
    all_states: set[PuzzleState] = {}

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
        initial_state = PuzzleState(initial)
        goal_state = PuzzleState(goal)

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

        context = {"nodes": nodes, "edges": edges}
        return render_template("index.html", **context)


@app.route('/cleaned', methods=['GET'])
def render_cleaned_state_space_tree():
    if request.method == 'GET':
        initial = [2, 8, 3, 1, 6, 4, 7, 0, 5]
        goal = [2, 0, 8, 1, 6, 3, 7, 5, 4]
        initial_state = PuzzleState(initial,0)
        goal_state = PuzzleState(goal,0)

        explored_states: set[PuzzleState] = cleaned_bfs(initial_state, goal_state)

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

        context = {"nodes": nodes, "edges": edges}
        return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(debug=True)
