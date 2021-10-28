import math
import random


EXPLORATION_CONSTANT = math.sqrt(2)


class MCTS:
    def __init__(self, root, game):
        self.root = root

    def mcts_step(self):
        leaf = self.root.select()
        rollout_leaf = leaf.expand()
        reward = rollout_leaf.rollout()
        rollout_leaf.backpropagate(reward)

    def simulate(self, n_simulations):
        for _ in range(n_simulations):
            self.mcts_step()

    def best_action(self):
        return max(self.root.edges.items(), key=lambda x: x[1]["N"])[0]

    def policy(self):
        return [(x[0], x[1]["N"]) for x in self.root.edges.items()]


class MCTSNode:

    def __init__(self, state, game, parent=None, action=None):
        self.state = state  
        self.game = game
        self.is_expanded = False
        self.edges = {}
        self.children = {}

        self.parent = parent
        self.action = action # action that leads to self, e.g. self.parent[action] = self
        self.is_root = True if self.parent is None else False

        self.available_actions = game["get_actions"](state)
        
    def add_child(self, action):
        new_state = self.game["make_action"](self.state, action)
        self.children[action] = MCTSNode(
                                    new_state,
                                    self.game,
                                    parent=self,
                                    action=action
                                    )

    def select(self):
        if self.is_expanded is False:
            return self
        else:
            N = sum(x["N"] for x in self.edges.values())
            selected_action = max(
                self.edges.items(),
                key=lambda x: calculate_action_value(x[1]["N"],
                                                x[1]["Q"],
                                                N,
                                                self.state[2,0,0])
            )[0]
            return self.children[selected_action].select()

    def expand(self):
        assert self.is_expanded is False

        if not self.game["is_terminal"](self.state):
            for action in self.available_actions:
                self.edges[action] = {
                    "N":1,
                    "W":0,
                    "Q":0
                }
                self.add_child(action)
            
            self.is_expanded = True

            action = random.choice(self.available_actions)
            return self.children[action]

        return self

    def rollout(self):
        assert self.is_expanded is False

        state = self.state

        while self.game["is_terminal"](state) == False:
            action = random.choice(self.game["get_actions"](state))
            state = self.game["make_action"](state, action)

        reward = self.game["get_reward"](state)
        return reward

    
    def backpropagate(self, reward):
        if not self.is_root:
            self.parent.edges[self.action]["N"] += 1
            self.parent.edges[self.action]["W"] += reward
            self.parent.edges[self.action]["Q"] += self.parent.edges[self.action]["W"]/self.parent.edges[self.action]["N"]
            
            self.parent.backpropagate(reward)

def calculate_action_value(n, Q, N, player):
    assert N >= 1
    assert n >= 1
    exploitation_term = player * Q 
    exploration_term = EXPLORATION_CONSTANT * math.sqrt(math.log(N)/n)
    return exploitation_term + exploration_term


