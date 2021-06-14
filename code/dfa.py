import random

class DFA:
    state_id = 0  # Keeps track of the discovery time of states for Tarjan's Algorithm.
    SCC = []  # Adds the SCC found from Tarjan's Algorithm.

    # DFA Initialization.
    def __init__(self, n_states, start, accept, reject, alphabet, transition):
        self.n_states = n_states  # Number of states - int
        self.start = start  # Starting state - int
        self.accept = accept  # Accepting states - List of int
        self.reject = reject  # Rejecting states - List of int
        self.alphabet = alphabet  # Alphabet - List of char
        self.transition = transition  # Transitions - Dictionary of Dictionaries

    # Implementation of Breadth First Search algorithm.
    def breadthFirstSearch(self, start):
        # Queue - a list to check immediate child of states first, then check their children (what state to check next).
        # Visited - a list to keep track of the states visited.
        queue, visited = [], []
        queue.append(start)
        visited.append(start)
        # Depth - a dictionary to store the depths or distance for each node from the start state.
        depth = {i: 0 for i in range(self.n_states)}
        while (len(queue) != 0):
            # Store current as a node from the start of the queue and remove it from the queue - FIFO.
            current = queue[0]
            queue.pop(0)
            # For each state transitioned from the the current state.
            for item, value in self.transition[current].items():
                # If the child was not visited, append it to the visited and queue, and update it's depth.
                if value not in visited:
                    visited.append(value)
                    queue.append(value)
                    # The depth the state is an increment of the depth of it's parent state.
                    depth[value] = depth[current] + 1
        # Get the maximum depth and the states with the maximum depth.
        max_depth = max(depth.values())
        maximums = [i for i,t_depth in depth.items() if t_depth == max_depth]
        return maximums, max_depth

    # Finds the depth of the DFA using the BFS algorithm.
    def getDepth(self):
        # Find the node(s) with longest path from the start state of the DFA.
        maximums, max_depth = self.breadthFirstSearch(self.start)
        maximum, final_depth = maximums, max_depth  # To store the max depth node and depth.
        # Find the max distance from each of the max depth states and update the max depth accordingly.
        for state in maximums:
            # Find the node with longest path from the previously found node of the DFA.
            temp_maximum, temp_depth = self.breadthFirstSearch(state)
            # Update the max depth and node if a larger dpeth value is found.
            if temp_depth > final_depth:
                maximum = temp_maximum
                final_depth = temp_depth
        # Displays the depth of the DFA.
        print("The number of states in the DFA is:", self.n_states)
        print("The depth of the DFA is:", final_depth)
        return final_depth

    # Finds all the states in the DFA that reach A from a given character c.
    def findStates(self, transition, c, A):
        found = []
        for state, values in transition.items():
            if (c in values.keys()) and (values[c] in A):
                found.append(state)
        return found

    # Implementation of Hopcroft's algorithm.
    def hopcroftsAlgorithm(self):
        # Partitions includes accepting and rejecting states..
        partitions = [self.accept, self.reject]
        # Distinguishers has only accepting states
        D = [self.accept]
        while len(D) != 0:
            # Remove a partition from the distinguishers.
            A = D.pop(0)
            # For each input in the DFA alphabet.
            for c in self.alphabet:
                # X is the states on which a transition on c will lead to a state in A.
                X = self.findStates(self.transition, c, A)
                # For each partition in the partitions.
                for i, part in enumerate(partitions[:]):
                    # If the set of states leading A and not leading to A are non empty.
                    intersection = list(set(X).intersection(part))
                    difference = list(set(part).difference(X))
                    if (len(intersection) != 0 and len(difference) != 0):
                        # Refine the partition further by replacing it with the two sets.
                        partitions.remove(part)
                        partitions.append(intersection)
                        partitions.append(difference)
                        # If the partition is found in the distinguishers.
                        if part in D:
                            # Replace the partition in distinguishers with the two sets.
                            D.remove(part)
                            D.append(intersection)
                            D.append(difference)
                        else:
                            # Append the intersection or difference depending on which is smaller.
                            if len(intersection) <= len(difference):
                                D.append(intersection)
                            else:
                                D.append(difference)
        # Returns the final partitions.
        return partitions

    # For updating the transitions by finding new transitions from a given state in a partition (A) to other partitions.
    def getNewTransition(self, A, partitions):
        # The old transition table.
        old_transitions = self.transition[A]
        # To store the transitions from A to other partitions.
        new_transitions = {}
        # For each character in the alphabet.
        for c in self.alphabet:
            # If transition exists.
            if c in old_transitions.keys():
                state = old_transitions[c]  # State that is reached from A using c.
                # Go through each new partition.
                for i, partition in enumerate(partitions):
                    # If the state is found in a partition, set the transition from A using c as it's index.
                    if state in partition:
                        new_transitions[c] = i
                        break
        return new_transitions

    # Minimizes the DFA using Hopcroft's then updates it's features by representing new partitions as states.
    def doMinimization(self):
        # Minimize the DFA using Hopcroft's algorithm.
        partitions = self.hopcroftsAlgorithm()
        # To store the new DFA features.
        new_transitions, new_accepting, new_rejecting, new_start = {}, [], [], self.start

        # For each partition.
        for i, partition in enumerate(partitions):
            # Check each state in the partition to check if there is a start state.
            if self.start in partition:
                new_start = i  # Setting the partition to a new start state.
            A = partition[0]  # Take a random state from the partition - (they should transition to the same partition).
            # Check whether the partition is an accepting or rejecting state.
            if (A in self.accept):
                new_accepting.append(i)
            else:
                new_rejecting.append(i)
            # Get the transitions from the current partition to other partitions.
            new_transitions[i] = self.getNewTransition(A, partitions)

        # Updating the states of the new minimized DFA.
        self.accept = new_accepting
        self.reject = new_rejecting
        self.start = new_start
        self.transition = new_transitions
        self.n_states = len(partitions)

    def checkDFA(self):
        string_len = 128 # Length of the strings
        string_n = 100 # Number of strings
        print("\n ------- Random String Classification  ------- ")
        # Iterating through string_n number of times or through each newly generated string.
        for i in range(string_n):
            current = self.start  # Test each string from the starting state.
            string = ''  # To append each character onto.
            # Iterating through a random number between 0 and string_len and generating a random character each time.
            for j in range(0, random.randint(0, string_len)):
                character = random.choice(self.alphabet) # Choosing a random character form the alphabet.
                string += character # Appending character to current string.
                # If transition with character doesnt exist, reject immediately.
                if character not in self.transition[current]:
                    current = None
                    break
                # Transition to the next state with the given character.
                current = self.transition[current][character]
            # Check if the state is in accepting states or rejecting states.
            if current in self.accept:
                print(string + ":: Accepting")
            else:
                print(string + ":: Rejecting")

    # A function that finds separated components in the DFA.
    def findSCC(self):
        # Reset Tarjan's Algorithm state_id and SCC class variables.
        self.state_id = 0
        self.SCC = []
        state_ids = [-1] * self.n_states  # Stores the state ids of a state when discovered.
        low_links = [-1] * self.n_states  # Stores the low link values.
        stack = []  # Stack to keep track of visited states.
        on_stack = [False] * self.n_states  # Keep track of what states are in the stack.

        # To ensure all states are explored in case of cycles.
        for i in range(self.n_states):
            # If the state isn't discovered yet, find it's SCC.
            if state_ids[i] == -1:
                self.tarjanDepthFirstSearch(i, on_stack, state_ids, low_links, stack)

        # Print the number of SCC found.
        print("\n --------Tarjan's Algorithm --------")
        print("The number of SCC is:", len(self.SCC))
        # Get largest and smallest SCC.
        largestSCC = max(self.SCC, key=len)
        smallestSCC = min(self.SCC, key=len)
        print("The largest SCC has size:", len(largestSCC))
        print("The smallest SCC has size:", len(smallestSCC))

    # Recursive DFS function with the implementation of Tarjan's algorithm to find SCC.
    def tarjanDepthFirstSearch(self, state, on_stack, state_ids, low_links, stack):
        # Upon discovering a state assign it an ID and low-link value.
        state_ids[state] = self.state_id
        low_links[state] = self.state_id
        self.state_id += 1
        # Append the state to the stack.
        on_stack[state] = True
        stack.append(state)

        # Iterate through the neighbouring states of the state.
        for child in self.transition[state].values():
            # If a child isn't discovered, recursively call function (DFS) to seek it's neighbours.
            if state_ids[child] == -1:
                self.tarjanDepthFirstSearch(child, on_stack, state_ids, low_links, stack)
            # After backtracking, update the low link value of the state if the child is on the stack.
            if on_stack[child]:
                low_links[state] = min(low_links[child], low_links[state])

        # After visiting all neighbours if current state is a head of a SCC.
        if low_links[state] == state_ids[state]:
            current = -1
            components = []  # To store the states of the SCC.
            # Pop the states from the stack till the head of the SCC is reached.
            while current != state:
                current = stack.pop()
                on_stack[current] = False
                components.append(current)  # Appending each state of the current SCC.
            self.SCC.append(components)  # Appending the SCC list to the class variable.