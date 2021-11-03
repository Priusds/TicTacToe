import random
import math
import torch


EXPLORATION_CONSTANT = 1000*math.sqrt(2)


class DMCTSTree:
    def __init__(self, state, model, game):
        self.root = DMCTSNode(state, model, game)
        self.policy = [1/(self.root.game["nActions"]) for i in range(self.root.game["nActions"])]

    def mcts_step(self):
        leaf = self.root.select()
        value = leaf.expand_evaluate()
        leaf.backup(value)

    def simulate(self, n_simulations, temperature=1):
        for _ in range(n_simulations):
            self.mcts_step()
        
        self.policy_update(temperature)

    def policy_update(self, temperature):
        # TODO: speed up
        self.policy = [self.root.edges[i]["N"]**(1/temperature) if i in self.root.edges.keys() else 0 for i in range(self.root.game["nActions"])]
        norm = sum(self.policy)
        for i in range(self.root.game["nActions"]):
            self.policy[i] = self.policy[i]/norm
       

    def best_action(self):
        return random.choices(range(self.root.game["nActions"]),self.policy)[0]
    


class DMCTSNode:
    def __init__(self, state, model, game, parent=None, action=None) -> None:
        self.is_expanded = False
        self.model = model
        self.state = state
        self.edges = {}
        self.children = {}
        self.game = game
        self.parent = parent
        self.action = action
        self.available_actions = game["get_actions"](state)


    def add_child(self, action):
        new_state = self.game["make_action"](self.state, action)
        self.children[action] = DMCTSNode(new_state, self.model, self.game, 
                                            parent=self, action=action)


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
                                                x[1]["P"],
                                                self.state[0,2,0,0].item())
            )[0]
            return self.children[selected_action].select()

    def expand_evaluate(self):
        assert self.is_expanded is False
        self.model.eval()

        p_v = torch.sigmoid(self.model(self.state))
        
        if not self.game["is_terminal"](self.state):
            for action in self.available_actions:
                self.edges[action] = {
                    "N":0,  # TODO Maybe start with N=1 -> first move depends on prior.
                    "W":0,
                    "Q":0,
                    "P":p_v[0,action].item()
                }
                self.add_child(action)
            
            self.is_expanded = True
        
            return p_v[0,-1].item()
        
        return self.game["get_reward"](self.state) # TODO try also with just returning v
    
    def backup(self, value):
        if not self.parent is None:
            self.parent.edges[self.action]["N"] += 1
            self.parent.edges[self.action]["W"] += value
            self.parent.edges[self.action]["Q"] += self.parent.edges[self.action]["W"]/self.parent.edges[self.action]["N"]
            
            self.parent.backup(value)

def calculate_action_value(n, Q, N, P, player):
    assert N >= 0, n >= 0
    exploitation_term = player * Q 
    exploration_term = EXPLORATION_CONSTANT * P * math.sqrt(N)/(1+n)
  #  print("expoit: ", exploitation_term)
  #  print("explore: ", exploration_term)
    return exploitation_term + exploration_term
