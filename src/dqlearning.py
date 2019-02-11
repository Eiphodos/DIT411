import os
import random
import torch
import time
import sys
import torch.nn as nn
import torch.optim as optim
import numpy as np
from game.wsgame import State
from collections import namedtuple



Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

def main(mode):
    cuda_available = torch.cuda.is_available()
    #cuda_available = False

    if mode == 'test':
        model = torch.load('trained_model/model.pth', map_location='cpu' if not cuda_available else None).eval()

        if cuda_available:
            model = model.cuda()
        test(model)


    elif mode == 'train':
        if not os.path.exists('trained_model/'):
            os.mkdir('trained_model/')

        model = NeuralNetwork()

        if cuda_available:
            model = model.cuda()

        model.apply(init_weights)
        start = time.time()
        train(model, start)


def test(model):
    state = State(64, 1, 1)
    cuda_available = torch.cuda.is_available()

    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    # Set initial action to 'do nothing' for all three wolves
    action[0] = 1
    action[5] = 1
    action[10] = 1

    #Get game grid and reward
    grid, reward = state.frame_step(action)

    #Convert to tensor
    tensor_data = torch.from_numpy(grid)
    if cuda_available:
        tensor_data = tensor_data.cuda()
    # Concatenate four last grids
    state = torch.cat((tensor_data, tensor_data, tensor_data, tensor_data)).unsqueeze(0)

    while reward != 1:
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if cuda_available:
            action = action.cuda()

        # Action
        action_index = torch.argmax(output)
        if cuda_available:
            action_index = action_index.cuda()
        action[action_index] = 1

        # State
        grid, reward = state.frame_step(action)
        tensor_data_1 = torch.from_numpy(grid)
        if cuda_available:
            tensor_data_1 = tensor_data_1.cuda()
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], tensor_data_1)).unsqueeze(0)

        # set state to be state_1
        state = state_1

def train(model, start):
    # define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # initialize mean squared error loss
    criterion = nn.MSELoss()

    # instantiate game
    game_state = State(4, 1, 1)

    # initialize replay memory
    replay_memory = []

    #Cuda
    cuda_available = torch.cuda.is_available()
    #cuda_available = False

    action = torch.zeros([model.n_actions], dtype=torch.float32)
    # Set initial action to 'do nothing' for all three wolves
    action[0] = 1

    #Get game grid and reward
    grid, reward = game_state.frame_step(action)

    #Convert to tensor
    tensor_data = torch.Tensor(grid)

    if cuda_available:
        tensor_data = tensor_data.cuda()
    # Concatenate four last grids
    state = tensor_data.unsqueeze(0).unsqueeze(0)

    # initialize epsilon value
    epsilon = model.init_epsilon
    iteration = 0

    epsilon_decrements = np.linspace(model.init_epsilon, model.fin_epsilon, model.iterations)

    # main infinite loop
    while iteration < model.iterations:
        # get output from the neural network
        output = model(state)[0]


        # initialize action
        action = torch.zeros([model.n_actions], dtype=torch.float32)
        if cuda_available:  # put on GPU if CUDA is available
            action = action.cuda()
        # epsilon greedy exploration wolf 1
        random_action = random.random() <= epsilon
        if random_action:
            print("Performed random action!")

        action_index_1 = [torch.randint(model.n_actions -1, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]


        if cuda_available:  # put on GPU if CUDA is available
            action_index_1 = action_index_1.cuda()


        action[action_index_1] = 1


        # State
        grid, reward = game_state.frame_step(action)
        tensor_data_1 = torch.Tensor(grid)
        if cuda_available:
            tensor_data_1 = tensor_data_1.cuda()
        state_1 = tensor_data_1.unsqueeze(0).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        # save transition to replay memory
        replay_memory.append((state, action, reward, state_1))

        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > model.memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # sample random minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        # unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if cuda_available:  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        # get output for the next state
        output_1_batch = model(state_1_batch)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i]
                                  for i in range(len(minibatch))))

        # extract Q-value
        q_value = torch.sum(model(state_batch) * action_batch, dim=1)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # calculate loss
        loss = criterion(q_value, y_batch)

        # do backward pass
        loss.backward()
        optimizer.step()

        # set state to be state_1
        state = state_1
        iteration += 1

        if iteration % 25000 == 0:
            torch.save(model, "trained_model/current_model_" + str(iteration) + ".pth")

        print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
              action_index_1.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
              np.max(output.cpu().detach().numpy()))

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.n_actions = 15
        self.gamma = 0.99
        self.fin_epsilon = 0.0001
        self.init_epsilon = 0.5
        self.iterations = 2000000
        self.memory_size = 10000
        self.minibatch_size = 32

        self.conv1 = nn.Conv2d(1, 8, 4, 3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(8, 8, 1, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(8, 32, 1, 2)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(32, 1)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(1, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

if __name__ == "__main__":
    main(sys.argv[1])