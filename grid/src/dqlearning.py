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
import argparse

from matplotlib import pyplot as plt
plt.style.use('bmh')


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

def main(args):

    parser = argparse.ArgumentParser()

    parser.add_argument('-mode', choices=['multi-test', 'single-test', 'multi-train', 'single-train'], help="Select a mode from either multi-test, single-test, multi-train, single-train")
    parser.add_argument('--iteration', help='Iteration number for the model you want to test, for example: 75000')

    args = parser.parse_args()
         

    # Game definitions used for testing and training
    grid_size = 7
    wolf_speed = 1
    sheep_speed = 1


    # Cuda was slower than running on a high end CPU. Possibly because matrixes are not large enough for for the GPU to be better at it.
    #cuda_available = torch.cuda.is_available()
    cuda_available = False

    # Testing multi agent system
    if args.mode == 'multi-test':
        if args.iteration:
            model1_str = 'trained_model/current_multi_model1_' + args.iteration + '.pth'
            model2_str = 'trained_model/current_multi_model2_' + args.iteration + '.pth'
            model3_str = 'trained_model/current_multi_model3_' + args.iteration + '.pth'

            model1 = torch.load(model1_str, map_location='cpu' if not cuda_available else None).eval()
            model2 = torch.load(model2_str, map_location='cpu' if not cuda_available else None).eval()
            model3 = torch.load(model3_str, map_location='cpu' if not cuda_available else None).eval()

            if cuda_available:
                model1 = model1.cuda()
                model2 = model2.cuda()
                model3 = model3.cuda()
            test_multi(model1, model2, model3, grid_size, wolf_speed, sheep_speed, cuda_available)
        else:
            raise ValueError('You need to supply an iteration number to test. Try -mode multi-test --iteration number')
    # Testing single agent system
    elif args.mode == 'single-test':
        if args.iteration:
            model_str = 'trained_model/current_single_model_' + args.iteration + '.pth'
            model = torch.load(model_str, map_location='cpu' if not cuda_available else None).eval()

            if cuda_available:
                model = model.cuda()
            test_single(model, grid_size, wolf_speed, sheep_speed, cuda_available)
        else:
            raise ValueError('You need to supply an iteration number to test. Try -mode single-test --iteration number')
    # Training multi agent system
    elif args.mode == 'multi-train':
        if not os.path.exists('trained_model/'):
            os.mkdir('trained_model/')

        model1 = NeuralNetwork(4)
        model2 = NeuralNetwork(4)
        model3 = NeuralNetwork(4)

        if cuda_available:
            model1 = model1.cuda()
            model2 = model2.cuda()
            model3 = model3.cuda()

        model1.apply(init_weights)
        model2.apply(init_weights)
        model3.apply(init_weights)
        start = time.time()
        train_networks(model1, model2, model3 , start, grid_size, wolf_speed, sheep_speed, cuda_available)

    # Training single agent system
    elif args.mode == 'single-train':
        if not os.path.exists('trained_model/'):
            os.mkdir('trained_model/')

        model = NeuralNetwork(64)

        if cuda_available:
            model = model.cuda()

        model.apply(init_weights)
        start = time.time()
        train_single_network(model, start, grid_size, wolf_speed, sheep_speed, cuda_available)


def test_multi(model1, model2, model3, grid_size, wolf_speed, sheep_speed, cuda):

    games_to_test = 10

    # Set cuda
    cuda_available = cuda

    # Instantiate game
    game_state = State(grid_size, wolf_speed, sheep_speed)

    # Set initial action for all three wolves
    action1 = torch.zeros([model1.n_actions], dtype=torch.float32)
    action2 = torch.zeros([model2.n_actions], dtype=torch.float32)
    action3 = torch.zeros([model3.n_actions], dtype=torch.float32)
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
    # Unsqueese to get the correct dimensons
    state = tensor_data.unsqueeze(0).unsqueeze(0)

    games = 0

    while games < games_to_test:

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

        if finished:
            games += 1

def test_single(model, grid_size, wolf_speed, sheep_speed, cuda):

    games_to_test = 10

    # Set cuda
    cuda_available = cuda

    # Instantiate game
    game_state = State(grid_size, wolf_speed, sheep_speed)

    action_wolf_1 = torch.zeros([4], dtype=torch.float32)
    action_wolf_2 = torch.zeros([4], dtype=torch.float32)
    action_wolf_3 = torch.zeros([4], dtype=torch.float32)
    # Set initial action
    action_wolf_1[0] = 1
    action_wolf_2[0] = 1
    action_wolf_3[0] = 1

    #Get game grid and reward
    grid, reward, finished = game_state.frame_step_single_reward(action_wolf_1, action_wolf_2, action_wolf_3)

    # Create drawing board and draw initial state
    window = Draw(grid_size, grid, False)
    window.update_window(grid)

    #Convert to tensor
    tensor_data = torch.Tensor(grid)

    if cuda_available:
        tensor_data = tensor_data.cuda()
    # Unsqueese to get the correct dimensons
    state = tensor_data.unsqueeze(0).unsqueeze(0)

    games = 0

    while games < games_to_test:

        # get output from the neural network for moving a wolf
        output = model(state)[0]

        # initialize actions
        action = torch.zeros([model.n_actions], dtype=torch.float32)
        if cuda_available:  # put on GPU if CUDA is available
            action = action.cuda()

        
    
        # Action #1
        action_index = torch.argmax(output)
        if cuda_available:
            action_index = action_index.cuda()    
        action[action_index] = 1

        action_wolf_1, action_wolf_2, action_wolf_3 = get_wolf_actions(action_index)

        # Update state
        grid, reward, finished = game_state.frame_step_single_reward(action_wolf_1, action_wolf_2, action_wolf_3)
        tensor_data_1 = torch.Tensor(grid)
        if cuda_available:
            tensor_data_1 = tensor_data_1.cuda()
        state_1 = tensor_data_1.unsqueeze(0).unsqueeze(0)

        #Draw new state
        window.update_window(grid)

        # set state to be state_1
        state = state_1
        if finished:
            games += 1


def train_networks(model1, model2, model3, start, grid_size, wolf_speed, sheep_speed, cuda):

    # Save Q-values for plotting
    q_history = []
    f = open ("q_history_multi.txt", "w+")

    # define Adam optimizer
    optimizer1 = optim.Adam(model1.parameters(), model1.learn_rate)
    optimizer2 = optim.Adam(model2.parameters(), model2.learn_rate)
    optimizer3 = optim.Adam(model3.parameters(), model3.learn_rate)

    # initialize replay memory
    replay_memory1 = []
    replay_memory2 = []
    replay_memory3 = []

    cuda_available = cuda

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
    game_state = State(grid_size, wolf_speed, sheep_speed)

    action1 = torch.zeros([model1.n_actions], dtype=torch.float32)
    action2 = torch.zeros([model2.n_actions], dtype=torch.float32)
    action3 = torch.zeros([model3.n_actions], dtype=torch.float32)
    # Set initial action for all three wolves
    action1[0] = 1
    action2[0] = 1
    action3[0] = 1

    #Get game grid and reward
    grid, reward1, reward2, reward3, finished = game_state.frame_step(action1, action2, action3)

    #Convert to tensor
    tensor_data = torch.Tensor(grid)

    if cuda_available:
        tensor_data = tensor_data.cuda()
    # Increase dimension on grid to fit shape for conv2d
    state = tensor_data.unsqueeze(0).unsqueeze(0)

    # Initialize iteration counter
    iteration = 0
    catches = 0
    avg_steps_per_catch = 0

    #Drawing while training
    enable_graphics = True
    if enable_graphics:
        window = Draw(grid_size, grid, True)
        window.update_window(grid)

    # All models have the same number of iterations so does not matter which one we are checking
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

        # set y_j to r_j for finished state, otherwise to r_j + gamma*max(Q)
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



        # A new tensor detached from the current graph
        y_batch_1 = y_batch_1.detach()
        y_batch_2 = y_batch_2.detach()
        y_batch_3 = y_batch_3.detach()

        # calculate loss
        loss1 = criterion(q_value_1, y_batch_1)
        loss2 = criterion(q_value_2, y_batch_2)
        loss3 = criterion(q_value_3, y_batch_3)

        # We reset gradients each pass
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        # do backward pass
        loss1.backward()
        loss2.backward()
        loss3.backward()

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

        # Save model every now and then
        if iteration % 25000 == 0:
            torch.save(model1, "trained_model/current_multi_model1_" +  str(iteration) +  ".pth")
            torch.save(model2, "trained_model/current_multi_model2_" +  str(iteration) +  ".pth")
            torch.save(model3, "trained_model/current_multi_model3_" +  str(iteration) +  ".pth")
        
        # Save Q-max
        q_max = np.max(output1.cpu().detach().numpy())
        q_history.append(q_max) 
        f.write("%f\n" % q_max)

        print("iteration:", iteration, "avg steps per catch: ", avg_steps_per_catch, "elapsed time:", time.time() - start, "epsilon:", epsilon1, "action:",
                action_index_1.cpu().detach().numpy(), "reward:", reward1.numpy()[0][0], "Q max:",
                q_max)
    plt.plot(q_history)
    plt.show()



def train_single_network(model, start, grid_size, wolf_speed, sheep_speed, cuda):

    # Save Q-values for plotting
    q_history = []
    f = open ("q_history.txt", "w+")

    enable_graphics = True

    # define Adam optimizer
    optimizer = optim.Adam(model.parameters(), model.learn_rate)

    # initialize replay memory
    replay_memory = []

    cuda_available = cuda

    # initialize epsilon value
    epsilon = model.init_epsilon
    epsilon_decrements = np.linspace(model.init_epsilon, model.fin_epsilon, model.iterations)

    # initialize mean squared error loss
    criterion = nn.MSELoss()

     # instantiate game
    game_state = State(grid_size, wolf_speed, sheep_speed)

    action_wolf_1 = torch.zeros([4], dtype=torch.float32)
    action_wolf_2 = torch.zeros([4], dtype=torch.float32)
    action_wolf_3 = torch.zeros([4], dtype=torch.float32)
    # Set initial action
    action_wolf_1[0] = 1
    action_wolf_2[0] = 1
    action_wolf_3[0] = 1

    #Get game grid and reward
    grid, reward, finished = game_state.frame_step_single_reward(action_wolf_1, action_wolf_2, action_wolf_3)

    #Convert to tensor
    tensor_data = torch.Tensor(grid)

    if cuda_available:
        tensor_data = tensor_data.cuda()
    # Increase dimensions of game grid to fit Conv2d
    state = tensor_data.unsqueeze(0).unsqueeze(0)

    # Initialize iteration counter
    iteration = 0
    catches = 0
    avg_steps_per_catch = 0

    #Drawing while training
    
    if enable_graphics:
        window = Draw(grid_size, grid, True)
        window.update_window(grid)

    while iteration < model.iterations:
        time_get_actions = time.time()

        # get output from the neural network
        output = model(state)[0]
 
        # initialize action
        action = torch.zeros([model.n_actions], dtype=torch.float32)
        if cuda_available:  # put on GPU if CUDA is available
            action = action.cuda()
        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        action_index = [torch.randint(model.n_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        if cuda_available:  # put on GPU if CUDA is available
            action_index= action_index.cuda()

        action[action_index] = 1

        action_wolf_1, action_wolf_2, action_wolf_3 = get_wolf_actions(action_index)

        print("Time to calculate actions: ",  time.time() - time_get_actions)

        time_move_state = time.time()

        #Get game grid and reward
        grid, reward, finished = game_state.frame_step_single_reward(action_wolf_1, action_wolf_2, action_wolf_3)
        tensor_data_1 = torch.Tensor(grid)
        if cuda_available:
            tensor_data_1 = tensor_data_1.cuda()
        state_1 = tensor_data_1.unsqueeze(0).unsqueeze(0)

        if enable_graphics:
            window.update_window(grid)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        # save transition to replay memory
        replay_memory.append((state, action, reward, state_1, finished))

        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > model.memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        print("Time to calculate state: ",  time.time() - time_move_state)

        time_calc_q = time.time()

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

        output_1_batch.volatile = False


        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                            else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                            for i in range(len(minibatch))))

        # extract Q-value
        q_value = torch.sum(model(state_batch) * action_batch, dim=1)

        print("Time to calculate q: ",  time.time() - time_calc_q)

        time_update_nn = time.time()

        # A new Tensor, detached from the current graph
        y_batch = y_batch.detach()

        # calculate loss
        loss = criterion(q_value, y_batch)

        # Reset gradients
        optimizer.zero_grad()

        # do backward pass
        loss.backward()
        optimizer.step()

        print("Time to update nn: ",  time.time() - time_update_nn)

        # set state to be state_1
        state = state_1

        # Update counters
        iteration += 1
        if (finished):
            catches += 1
            avg_steps_per_catch = iteration / catches

        # Save model once in a while
        if iteration % 25000 == 0:
            torch.save(model, "trained_model/current_single_model_" +  str(iteration) +  ".pth")

        # Save Q-max 
        q_max = np.max(output.cpu().detach().numpy())
        q_history.append(q_max) 
        f.write("%f\n" % q_max)

        print("iteration:", iteration, "avg steps per catch: ", avg_steps_per_catch, "elapsed time:", time.time() - start, "time per iteration: ", (time.time() - start) / iteration, "epsilon:", epsilon, "action:",
                action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
                q_max)
    plt.plot(q_history)
    plt.show()

   
        

def get_wolf_actions(action_index):

    action_wolf_1 = torch.zeros([4], dtype=torch.float32)
    action_wolf_2 = torch.zeros([4], dtype=torch.float32)
    action_wolf_3 = torch.zeros([4], dtype=torch.float32)

    index = action_index.cpu().numpy().astype(int)
    # See lookup table at the end of this file for the basis to these calculations
    wolf_1_index = index // 16
    wolf_2_index = ( index // 4 ) % 4
    wolf_3_index = index % 4 

    action_wolf_1[wolf_1_index] = 1
    action_wolf_2[wolf_2_index] = 1
    action_wolf_3[wolf_3_index] = 1

    return action_wolf_1, action_wolf_2, action_wolf_3


class NeuralNetwork(nn.Module):

    def __init__(self, n_actions):
        super(NeuralNetwork, self).__init__()

        self.n_actions = n_actions
        # To incentivize early catches we have tried gamma between 0.9 and 0.99
        self.gamma = 0.99
        self.fin_epsilon = 0.001
        self.init_epsilon = 0.15
        self.iterations = 500000
        self.memory_size = 10000
        self.minibatch_size = 32
        self.learn_rate = 1e-3

        # Channels = 1
        # Filters = 4
        # Filter size = 3x3
        # Stride = 1
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.relu1 = nn.ReLU(inplace=True)
        # Channels = 4
        # Filters = 8
        # Filter size = 3x3
        # Stride = 1
        self.conv2 = nn.Conv2d(4, 8, 3, 1)
        self.relu2 = nn.ReLU(inplace=True)
        # Channels = 8
        # Filters = 16
        # Filter size = 3x3
        # Stride = 1
        self.conv3 = nn.Conv2d(8, 16, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(16, 8)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(8, self.n_actions)
        

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





# The action space for the single agent system
# As an example, if action_index = 24
# Then wolf_1 should move south, wolf_2 should move east and wolf_3 should move west.
        # wolf directions
        # 0 - WWW
        # 1 - WWS
        # 2 - WWE
        # 3 - WWN
        # 4 - WSW
        # 5 - WSS
        # 6 - WSE
        # 7 - WSN
        # 8 - WEW
        # 9 - WES
        # 10 - WEE
        # 11 - WEN
        # 12 - WNW
        # 13 - WNS
        # 14 - WNE
        # 15 - WNN
        # 16 - SWW
        # 17 - SWS
        # 18 - SWE
        # 19 - SWN
        # 20 - SSW
        # 21 - SSS
        # 22 - SSE
        # 23 - SSN
        # 24 - SEW
        # 25 - SES
        # 26 - SEE
        # 27 - SEN
        # 28 - SNW
        # 29 - SNS
        # 30 - SNE
        # 31 - SNN
        # 32 - EWW
        # 33 - EWS
        # 34 - EWE
        # 35 - EWN
        # 36 - ESW
        # 37 - ESS
        # 38 - ESE
        # 39 - ESN
        # 40 - EEW
        # 41 - EES
        # 42 - EEE
        # 43 - EEN
        # 44 - ENW
        # 45 - ENS
        # 46 - ENE
        # 47 - ENN
        # 48 - NWW
        # 49 - NWS
        # 50 - NWE
        # 51 - NWN
        # 52 - NSW
        # 53 - NSS
        # 54 - NSE
        # 55 - NSN
        # 56 - NEW
        # 57 - NES
        # 58 - NEE
        # 59 - NEN
        # 60 - NNW
        # 61 - NNS
        # 62 - NNE
        # 63 - NNN