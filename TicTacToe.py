import random
import numpy as np
import torch


action_to_index = {
    0:(0,0),
    1:(0,1),
    2:(0,2),

    3:(1,0),
    4:(1,1),
    5:(1,2),

    6:(2,0),
    7:(2,1),
    8:(2,2),
}


def initial_state():
    return np.concatenate((np.zeros(shape=(2,3,3), dtype=np.float32),
                            (2*random.randint(0, 1)-1)*np.ones(shape=(1,3,3), dtype=np.float32)),
                            axis = 0)

def initial_state_torch():
    return torch.from_numpy(initial_state()).unsqueeze(dim=0)


def get_actions(state):
    return [idx for idx in range(9) if 
                            state[0,:,:][action_to_index[idx]] == 
                            state[1,:,:][action_to_index[idx]] == 
                            0]
def get_actions_torch(state):
    return [idx for idx in range(9) if 
                            state[0,0,:,:][action_to_index[idx]] == 
                            state[0,1,:,:][action_to_index[idx]] == 
                            0]

def make_action(state, action):
    new_state = np.copy(state)
    new_state[2,:,:] = -1*new_state[2,:,:]
    i = 0 if state[2,0,0] == 1 else 1
    new_state[i,:,:][action_to_index[action]] = 1
    
    return new_state

def make_action_torch(state, action):
    new_state = state.clone()
    new_state[0,2,:,:] = -1*new_state[0,2,:,:]
    i = 0 if state[0,2,0,0] == 1 else 1
    new_state[0,i,:,:][action_to_index[action]] = 1
    
    return new_state

def get_reward(state):
    board = state[0,:,:] - state[1,:,:]
    rows = board.sum(axis=1)
    cols = board.sum(axis=0)
    diag1 = np.trace(board)
    diag2 = np.trace(np.flip(board, axis=1))

    if rows.max() == 3 or cols.max() == 3 or diag1 == 3 or diag2 == 3:
        return 1
    elif rows.min()==-3 or cols.min()==-3 or diag2==-3 or diag1==-3:
        return -1
    else:
        return 0

def get_reward_torch(state):
    board = state[0,0,:,:] - state[0,1,:,:]
    rows = board.sum(axis=1)
    cols = board.sum(axis=0)
    diag1 = board.trace().item()
    diag2 = (board[2,0]+board[1,1]+board[0,2]).item()

    if rows.max() == 3 or cols.max() == 3 or diag1 == 3 or diag2 == 3:
        return 1
    elif rows.min()==-3 or cols.min()==-3 or diag2==-3 or diag1==-3:
        return -1
    else:
        return 0

def is_terminal(state):
    if len(get_actions(state)) == 0:
        return True
    else:
        if get_reward(state) != 0:
            return True
    return False

def is_terminal_torch(state):
    if len(get_actions_torch(state)) == 0:
        return True
    else:
        if get_reward_torch(state) != 0:
            return True
    return False

def state_to_string(state):
    symbol = {
        1: "X",
        -1: "O",
        0: " "
    }
    board = state[0,:,:] - state[1,:,:]
    string = ""
    for i in range(3):
        row = board[i,:]
        string += symbol[row[0]]
        string += "|"+symbol[row[1]]
        string += ("|"+symbol[row[2]]+"\n")

    return string


TicTacToe = {
    "nActions":9,
    "get_actions":get_actions,
    "make_action":make_action,
    "is_terminal":is_terminal,
    "initial_state":initial_state(),
    "nPositions":9,
    "get_reward":get_reward,
    "action_to_idx":action_to_index,
    "state_to_string":state_to_string
}

TicTacToeTorch = {
    "nActions":9,
    "get_actions":get_actions_torch,
    "make_action":make_action_torch,
    "is_terminal":is_terminal_torch,
    "initial_state":initial_state_torch(),
    "nPositions":9,
    "get_reward":get_reward_torch,
    "action_to_idx":action_to_index,
    "state_to_string":state_to_string
}
