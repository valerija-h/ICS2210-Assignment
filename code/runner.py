from dfa import DFA  # Importing the DFA class from dfa.py
from random import *

# Make a dictionary where key=states and values=dictionary (where key=characters and value=state you will reach).
def createDFSA():
    alphabet = ['a', 'b']
    # Find a random number between 16-64.
    no_states = randint(16, 64)
    # Create a dictionary with no_states where each state has a and b leading to a random two other states
    transitions = {}
    # Go through each state and create a transition two a random another state.
    for i in range(0, no_states):
        state1 = randint(0, no_states - 1)
        state2 = randint(0, no_states - 1)
        transitions[i] = {'a': state1, 'b': state2}
    # Choose random number for accepting or rejecting, I chose 20% to be accepting states and sorted them in order.
    accepting = sample(transitions.keys(), k=(round(.2 * no_states)))
    accepting.sort()
    # Choose a random starting state
    initial = randint(0, no_states - 1)
    # Get rejecting states
    all_states = [i for i in range(0, no_states)]
    rejecting = list(set(all_states).symmetric_difference(accepting))
    return DFA(no_states, initial, accepting, rejecting, alphabet, transitions)

def main():
    DFA1 = createDFSA()
    # Question no.2
    DFA1.getDepth()
    # Question no.3
    DFA1.doMinimization()
    # Question no.4
    DFA1.getDepth()
    # Question no.5
    DFA1.checkDFA()
    # Question no.6
    DFA1.findSCC()

main()