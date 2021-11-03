from torch.utils import data
from deep_mcts.nn import Net
import torch
from deep_mcts.deep_mcts import  DMCTSTree
from TicTacToe import TicTacToeTorch
from torch.utils.data import Dataset, DataLoader
import random
import torch.optim as optim
import time


def zero_decay(n_moves):
    return 1

def step_decay(n_moves):
    return 1 if (0<=n_moves<=2) else (0.5 if (3<=n_moves<=5) else 0.1)

def generate_data(n_games, n_simulations, game, model, decay):

    list_states = []
    list_policies = []
    list_rewards = []

    for _ in range(n_games):

        state = game["initial_state"]
        tree = DMCTSTree(state, model, game)

        n_moves = 0
        while not game["is_terminal"](state):
            
            tree.simulate(n_simulations, decay(n_moves))

            list_states.append(state)
            list_policies.append(tree.policy)

            action = tree.best_action()
            state = game["make_action"](state, action)

            tree.root = tree.root.children[action]
            tree.root.parent = None

            n_moves += 1
        
        reward = game["get_reward"](state)

        list_rewards = list_rewards + [reward]*n_moves
    
    return list_states, list_policies, list_rewards


class GamesDataSet(Dataset):
    def __init__(self, list_states, list_policies, list_rewards, transform=None):
        self.list_states = list_states
        self.list_policies = list_policies
        self.list_rewards = list_rewards
        self.transform = transform

    def __len__(self):
        return len(self.list_rewards)
    
    def __getitem__(self, idx):
        # TODO add transform
        state = self.list_states[idx][0,:,:,:]
        policy = torch.tensor(self.list_policies[idx])
        reward = torch.tensor(self.list_rewards[idx], dtype=torch.float32)
        
        if self.transform:
            bernoulli = random.randint(0,1)
            
            if bernoulli == 1:
                new_state = state
                new_state[0,:,:] = state[1,:,:]
                new_state[1,:,:] = state[0,:,:]
                new_state[2,:,:] = -1*state[2,:,:]

                reward = -1*reward

                state = new_state
            
        return state, policy, reward


N_ROUNDS = 10
N_GAMES = 10
N_SIMULATIONS = 200
N_EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 20

N_FEATURES = 20
N_RESIDUALS = 5
MODEL_PATH = "./models/"

model = Net(N_FEATURES, N_RESIDUALS, 9) 

print("Model has ", model.trainable_params())


for i in range(N_ROUNDS):
        
    initial_time = time.time()
    list_states, list_policies, list_rewards = generate_data(N_GAMES, N_SIMULATIONS, TicTacToeTorch, model, step_decay)

    print("Time for generating games: ", time.time()-initial_time)

    dataset = GamesDataSet(list_states, list_policies, list_rewards, transform=None)

    print("Dataset has ", len(dataset), " entries.")

    training_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    #lossCE =  torch.nn.CrossEntropyLoss()


    def lossCE(predicted, target):
        c = torch.log(torch.exp(predicted).sum(dim=1))
        return -((predicted * target).sum(dim=1) - c * target.sum(dim=1)).mean()

    lossL2 = torch.nn.MSELoss()

    model.train()
    initial_time = time.time()
    for epoch in range(N_EPOCHS):

        batch_loss = 0
        for data in training_loader:
            state, target_policy, target_reward = data

            predictions = model(state)

            loss1 = lossCE(predictions[:,0:-1], target_policy)
            loss2 = lossL2(predictions[:,-1], target_reward)
            

            loss_total = loss1 + loss2 
            loss_total.backward()


            optimizer.zero_grad()
            optimizer.step()

            batch_loss += loss_total.item()

        if epoch%10 == 0:
            print(f"Batch {epoch} has error {batch_loss}")

    print("Training time: ", time.time()-initial_time)
torch.save(model.state_dict(), MODEL_PATH + "net.pth")