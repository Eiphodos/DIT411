import os
import random
import torch
import time
import sys
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from game.wsgame import State
from draw import Draw
from collections import namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

def main(mode):
    #cuda_available = torch.cuda.is_available()
    cuda_available = False

    if mode == 'test':
        model1 = torch.load('trained_model/current_model1_25000.pth', map_location='cpu' if not cuda_available else None).eval()
        model2 = torch.load('trained_model/current_model2_25000.pth', map_location='cpu' if not cuda_available else None).eval()
        model3 = torch.load('trained_model/current_model3_25000.pth', map_location='cpu' if not cuda_available else None).eval()

        if cuda_available:
            model1 = model1.cuda()
            model2 = model2.cuda()
            model3 = model3.cuda()
        test(model1, model2, model3)


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


def test(model1, model2, model3):

    grid_size = 5
    wolf_speed = 1
    sheep_speed = 1

    state = State(grid_size, wolf_speed, sheep_speed)

    #cuda_available = torch.cuda.is_available()
    cuda_available = False

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

    # Create drawing board and draw initial state
    window = Draw(grid_size, grid, False)
    window.update_window(grid)

    #Convert to tensor
    tensor_data = torch.Tensor(grid)

    if cuda_available:
        tensor_data = tensor_data.cuda()
    # Concatenate four last grids
    state = tensor_data.unsqueeze(0).unsqueeze(0)

    while not finished:

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

        # Action
        action_index1 = torch.argmax(output1)
        action_index2 = torch.argmax(output2)
        action_index3 = torch.argmax(output3)
        if cuda_available:
            action_index1 = action_index1.cuda()
            action_index2 = action_index2.cuda()
            action_index3 = action_index3.cuda()
        
        action1[action_index1] = 1
        action2[action_index2] = 1
        action3[action_index3] = 1

        # State
        grid, reward1, reward2, reward3, finished = game_state.frame_step(action1, action2, action3)
        tensor_data_1 = torch.Tensor(grid)
        if cuda_available:
            tensor_data_1 = tensor_data_1.cuda()
        state_1 = tensor_data_1.unsqueeze(0).unsqueeze(0)

        #Draw new state
        window.update_window(grid)

        # set state to be state_1
        state = state_1

def train_networks(model1, model2, model3, start):

    grid_size = 5

    # define Adam optimizer
    optimizer1 = optim.Adam(model1.parameters(), model1.learn_rate)
    optimizer2 = optim.Adam(model2.parameters(), model2.learn_rate)
    optimizer3 = optim.Adam(model3.parameters(), model3.learn_rate)

    # initialize replay memory
    replay_memory1 = []
    replay_memory2 = []
    replay_memory3 = []

    #Cuda
    #cuda_available = torch.cuda.is_available()
    cuda_available = False

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
    game_state = State(grid_size, 1, 1)

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
    catches = 0
    avg_steps_per_catch = 0

    #Drawing while training
    enable_graphics = True
    if enable_graphics:
        window = Draw(grid_size, grid, True)
        window.update_window(grid)

    while iteration < model1.iterations:
        time_get_actions = time.time()

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

        print("Time to calculate actions: ",  time.time() - time_get_actions)

        time_move_state = time.time()
        # State
        grid, reward1, reward2, reward3, finished = game_state.frame_step(action1, action2, action3)
        tensor_data_1 = torch.Tensor(grid)
        if cuda_available:
            tensor_data_1 = tensor_data_1.cuda()
        state_1 = tensor_data_1.unsqueeze(0).unsqueeze(0)

        if enable_graphics:
            window.update_window(grid)

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

        print("Time to calculate state: ",  time.time() - time_move_state)

        time_calc_q = time.time()

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

        print("Time to calculate q: ",  time.time() - time_calc_q)

        time_update_nn = time.time()



        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch_1 = y_batch_1.detach()
        y_batch_2 = y_batch_2.detach()
        y_batch_3 = y_batch_3.detach()

        # calculate loss
        loss1 = criterion(q_value_1, y_batch_1)
        loss2 = criterion(q_value_2, y_batch_2)
        loss3 = criterion(q_value_3, y_batch_3)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        # do backward pass
        loss1.backward()
        loss2.backward()
        loss3.backward()

        for param in model1.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in model2.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in model3.parameters():
            param.grad.data.clamp_(-1, 1)

        optimizer1.step()
        optimizer2.step()
        optimizer3.step()

        print("Time to update nn: ",  time.time() - time_update_nn)
        # set state to be state_1
        state = state_1
        iteration += 1
        if (finished):
            catches += 1
            avg_steps_per_catch = iteration / catches

        if iteration % 25000 == 0:
            torch.save(model1, "trained_model/current_model1_" +  str(iteration) +  ".pth")
            torch.save(model2, "trained_model/current_model2_" +  str(iteration) +  ".pth")
            torch.save(model3, "trained_model/current_model3_" +  str(iteration) +  ".pth")

        print("iteration:", iteration, "avg steps per catch: ", avg_steps_per_catch, "elapsed time:", time.time() - start, "epsilon:", epsilon1, "action:",
                action_index_1.cpu().detach().numpy(), "reward:", reward1.numpy()[0][0], "Q max:",
                np.max(output1.cpu().detach().numpy()))




class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.n_actions = 5
        self.gamma = 0.99
        self.fin_epsilon = 0.001
        self.init_epsilon = 0.9
        self.iterations = 50000
        self.memory_size = 10000
        self.minibatch_size = 64
        self.epochs = 1000
        self.learn_rate = 1e-5
 
        self.conv1 = nn.Conv2d(1, 8, (2,2))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(8, 16, (2,2))
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 16, (2,2))
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(64, 1)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(1, 1)
        
        
        #self.conv1 = nn.Conv2d(1, 8, 4, 3)
        #self.relu1 = nn.ReLU(inplace=True)
        #self.conv2 = nn.Conv2d(8, 8, 1, 2)
        #self.relu2 = nn.ReLU(inplace=True)
        #self.conv3 = nn.Conv2d(8, 32, 1, 2)
        #self.relu3 = nn.ReLU(inplace=True)
        #self.fc4 = nn.Linear(32, 1)
        #self.relu4 = nn.ReLU(inplace=True)
        #self.fc5 = nn.Linear(1, 1)

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