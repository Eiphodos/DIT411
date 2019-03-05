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

        model1 = NeuralNetwork()
        model2 = NeuralNetwork()
        model3 = NeuralNetwork()

        if cuda_available:
            model1 = model1.cuda()
            model2 = model2.cuda()
            model3 = model3.cuda()

        model1.apply(init_weights)
        model2.apply(init_weights)
        model3.apply(init_weights)
        start = time.time()
        train_networks(model1, model2, model3 , start)


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

def train_networks(model1, model2, model3, start):

    # define Adam optimizer
    optimizer1 = optim.Adam(model1.parameters(), model1.learn_rate)
    optimizer2 = optim.Adam(model2.parameters(), model2.learn_rate)
    optimizer3 = optim.Adam(model3.parameters(), model3.learn_rate)

    # initialize replay memory
    replay_memory1 = []
    replay_memory2 = []
    replay_memory3 = []

    #Cuda
    cuda_available = torch.cuda.is_available()

    # initialize epsilon value
    epsilon1 = model1.init_epsilon
    epsilon_decrements1 = np.linspace(model1.init_epsilon, model1.fin_epsilon, model1.iterations)
    epsilon2 = model2.init_epsilon
    epsilon_decrements2 = np.linspace(model2.init_epsilon, model2.fin_epsilon, model2.iterations)
    epsilon3 = model3.init_epsilon
    epsilon_decrements3 = np.linspace(model3.init_epsilon, model3.fin_epsilon, model3.iterations)

    # initialize mean squared error loss
    criterion = nn.MSELoss()

     # instantiate game
    game_state = State(5, 1, 1)

    action1 = torch.zeros([model1.n_actions], dtype=torch.float32)
    action2 = torch.zeros([model2.n_actions], dtype=torch.float32)
    action3 = torch.zeros([model3.n_actions], dtype=torch.float32)
    # Set initial action to 'do nothing' for all three wolves
    action1[0] = 1
    action2[0] = 1
    action3[0] = 1

    #Get game grid and reward
    grid, reward1, reward2, reward3, finished = game_state.frame_step(action1, action2, action3)

    #Convert to tensor
    tensor_data = torch.Tensor(grid)

    if cuda_available:
        tensor_data = tensor_data.cuda()
    # Concatenate four last grids
    state = tensor_data.unsqueeze(0).unsqueeze(0)

    # Initialize iterion counter for this epoch
    iteration = 0

    while iteration < model1.iterations:

        # get output from the neural network
        output1 = model1(state)[0]
        output2 = model2(state)[0]
        output3 = model3(state)[0]

        # initialize actions
        action1 = torch.zeros([model1.n_actions], dtype=torch.float32)
        action2 = torch.zeros([model2.n_actions], dtype=torch.float32)
        action3 = torch.zeros([model3.n_actions], dtype=torch.float32)
        if cuda_available:  # put on GPU if CUDA is available
            action1 = action1.cuda()
            action2 = action2.cuda()
            action3 = action3.cuda()
        # epsilon greedy exploration wolf 1
        random_action1 = random.random() <= epsilon1
        action_index_1 = [torch.randint(model1.n_actions, torch.Size([]), dtype=torch.int)
                        if random_action1
                        else torch.argmax(output1)][0]

        # epsilon greedy exploration wolf 2
        random_action2 = random.random() <= epsilon2
        action_index_2 = [torch.randint(model2.n_actions, torch.Size([]), dtype=torch.int)
                        if random_action2
                        else torch.argmax(output2)][0]

        # epsilon greedy exploration wolf 3
        random_action3 = random.random() <= epsilon3
        action_index_3 = [torch.randint(model3.n_actions, torch.Size([]), dtype=torch.int)
                        if random_action3
                        else torch.argmax(output3)][0]


        if cuda_available:  # put on GPU if CUDA is available
            action_index_1 = action_index_1.cuda()
            action_index_2 = action_index_2.cuda()
            action_index_3 = action_index_3.cuda()


        action1[action_index_1] = 1
        action2[action_index_2] = 1
        action3[action_index_3] = 1

        # State
        grid, reward1, reward2, reward3, finished = game_state.frame_step(action1, action2, action3)
        tensor_data_1 = torch.Tensor(grid)
        if cuda_available:
            tensor_data_1 = tensor_data_1.cuda()
        state_1 = tensor_data_1.unsqueeze(0).unsqueeze(0)

        action1 = action1.unsqueeze(0)
        action2 = action2.unsqueeze(0)
        action3 = action3.unsqueeze(0)
        reward1 = torch.from_numpy(np.array([reward1], dtype=np.float32)).unsqueeze(0)
        reward2 = torch.from_numpy(np.array([reward2], dtype=np.float32)).unsqueeze(0)
        reward3 = torch.from_numpy(np.array([reward3], dtype=np.float32)).unsqueeze(0)

        # save transition to replay memory
        replay_memory1.append((state, action1, reward1, state_1, finished))
        replay_memory2.append((state, action2, reward2, state_1, finished))
        replay_memory3.append((state, action3, reward3, state_1, finished))

        # if replay memory is full, remove the oldest transition
        if len(replay_memory1) > model1.memory_size:
            replay_memory1.pop(0)
        if len(replay_memory2) > model2.memory_size:
            replay_memory2.pop(0)
        if len(replay_memory3) > model3.memory_size:
            replay_memory3.pop(0)


        # epsilon annealing
        epsilon1 = epsilon_decrements1[iteration]
        epsilon2 = epsilon_decrements2[iteration]
        epsilon3 = epsilon_decrements3[iteration]

        # sample random minibatch
        minibatch1 = random.sample(replay_memory1, min(len(replay_memory1), model1.minibatch_size))
        minibatch2 = random.sample(replay_memory2, min(len(replay_memory2), model2.minibatch_size))
        minibatch3 = random.sample(replay_memory3, min(len(replay_memory3), model3.minibatch_size))


        # unpack minibatch 1
        state_batch1 = torch.cat(tuple(d[0] for d in minibatch1))
        action_batch1 = torch.cat(tuple(d[1] for d in minibatch1))
        reward_batch1 = torch.cat(tuple(d[2] for d in minibatch1))
        state_1_batch1 = torch.cat(tuple(d[3] for d in minibatch1))

        # unpack minibatch 2
        state_batch2 = torch.cat(tuple(d[0] for d in minibatch2))
        action_batch2 = torch.cat(tuple(d[1] for d in minibatch2))
        reward_batch2 = torch.cat(tuple(d[2] for d in minibatch2))
        state_1_batch2 = torch.cat(tuple(d[3] for d in minibatch2))

        # unpack minibatch 3
        state_batch3 = torch.cat(tuple(d[0] for d in minibatch3))
        action_batch3 = torch.cat(tuple(d[1] for d in minibatch3))
        reward_batch3 = torch.cat(tuple(d[2] for d in minibatch3))
        state_1_batch3 = torch.cat(tuple(d[3] for d in minibatch3))

        if cuda_available:  # put on GPU if CUDA is available
            state_batch1 = state_batch1.cuda()
            state_batch2 = state_batch2.cuda()
            state_batch3 = state_batch3.cuda()
            action_batch1 = action_batch1.cuda()
            action_batch2 = action_batch2.cuda()
            action_batch3 = action_batch3.cuda()
            reward_batch1 = reward_batch1.cuda()
            reward_batch2 = reward_batch2.cuda()
            reward_batch3 = reward_batch3.cuda()
            state_1_batch1 = state_1_batch1.cuda()
            state_1_batch2 = state_1_batch2.cuda()
            state_1_batch3 = state_1_batch3.cuda()

        # get output for the next state
        output_1_batch1 = model1(state_1_batch1)
        output_1_batch2 = model2(state_1_batch2)
        output_1_batch3 = model3(state_1_batch3)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch_1 = torch.cat(tuple(reward_batch1[i] if minibatch1[i][4]
                            else reward_batch1[i] + model1.gamma * torch.max(output_1_batch1[i])
                            for i in range(len(minibatch1))))
        y_batch_2 = torch.cat(tuple(reward_batch2[i] if minibatch2[i][4]
                            else reward_batch2[i] + model2.gamma * torch.max(output_1_batch2[i])
                            for i in range(len(minibatch2))))
        y_batch_3 = torch.cat(tuple(reward_batch3[i] if minibatch3[i][4]
                            else reward_batch3[i] + model3.gamma * torch.max(output_1_batch3[i])
                            for i in range(len(minibatch3))))

        # extract Q-value
        q_value_1 = torch.sum(model1(state_batch1) * action_batch1, dim=1)
        q_value_2 = torch.sum(model2(state_batch2) * action_batch2, dim=1)
        q_value_3 = torch.sum(model3(state_batch3) * action_batch3, dim=1)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch_1 = y_batch_1.detach()
        y_batch_2 = y_batch_2.detach()
        y_batch_3 = y_batch_3.detach()

        # calculate loss
        loss1 = criterion(q_value_1, y_batch_1)
        loss2 = criterion(q_value_2, y_batch_2)
        loss3 = criterion(q_value_3, y_batch_3)

        # do backward pass
        loss1.backward()
        loss2.backward()
        loss3.backward()
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()

        # set state to be state_1
        state = state_1
        iteration += 1

        if iteration % 25000 == 0:
            torch.save(model1, "pretrained_model/current_model1_" + str(iteration) + ".pth")
            torch.save(model2, "pretrained_model/current_model2_" + str(iteration) + ".pth")
            torch.save(model3, "pretrained_model/current_model3_" + str(iteration) + ".pth")

        print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon1, "action:",
                action_index_1.cpu().detach().numpy(), "reward:", reward1.numpy()[0][0], "Q max:",
                np.max(output1.cpu().detach().numpy()))



# Training a single network, not currently used
def train(model1, model2, model3, start):
    # define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # initialize replay memory
    replay_memory = []

    #Cuda
    cuda_available = torch.cuda.is_available()
    #cuda_available = False

    # initialize epochs
    epoch = 0
    tot_iterations = 0

     # initialize epsilon value
    epsilon = model.init_epsilon
    epsilon_decrements = np.linspace(model.init_epsilon, model.fin_epsilon, model.iterations)

    # main infinite loop
    while epoch < model.epochs:

        # initialize mean squared error loss
        criterion = nn.MSELoss()

        # instantiate game
        game_state = State(10, 1, 1)

        action = torch.zeros([model.n_actions], dtype=torch.float32)
        # Set initial action to 'do nothing' for all three wolves
        action[0] = 1

        #Get game grid and reward
        grid, reward, finished = game_state.frame_step(action)

        #Convert to tensor
        tensor_data = torch.Tensor(grid)

        if cuda_available:
            tensor_data = tensor_data.cuda()
        # Concatenate four last grids
        state = tensor_data.unsqueeze(0).unsqueeze(0)

        # Initialize iterion counter for this epoch
        iteration = 0
        
        while (not finished):

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
            grid, reward, finished = game_state.frame_step(action)
            tensor_data_1 = torch.Tensor(grid)
            if cuda_available:
                tensor_data_1 = tensor_data_1.cuda()
            state_1 = tensor_data_1.unsqueeze(0).unsqueeze(0)

            action = action.unsqueeze(0)
            reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

            # save transition to replay memory
            replay_memory.append((state, action, reward, state_1, finished))

            # if replay memory is full, remove the oldest transition
            if len(replay_memory) > model.memory_size:
                replay_memory.pop(0)

            # epsilon annealing
            epsilon = epsilon_decrements[tot_iterations]

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
            y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                              else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
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
            tot_iterations += 1

            if iteration % 25000 == 0:
                torch.save(model, "trained_model/current_model_" + str(iteration) + ".pth")

            print("Epoch: " + str(epoch) + " iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
                  action_index_1.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
              np.max(output.cpu().detach().numpy()))
        epoch += 1

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.n_actions = 5
        self.gamma = 0.99
        self.fin_epsilon = 0.0001
        self.init_epsilon = 0.5
        self.iterations = 300000
        self.memory_size = 100000
        self.minibatch_size = 32
        self.epochs = 1000
        self.learn_rate = 1e-6

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