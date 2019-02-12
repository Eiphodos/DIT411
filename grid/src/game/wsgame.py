import numpy as np
import random

class State:
    def __init__(self, gridsize, wspeed, sspeed):
        #Free position = 0, wolf = 1, sheep = 2
        self.grid_size = gridsize
        self.grid = np.zeros((gridsize, gridsize), dtype=np.int)
        self.n_wolves = 3

        if (self.grid_size * self.grid_size < 9 ):
            raise ValueError('The grid is too small!')

        self.wolf_1_pos = (0, 0)
        self.grid[0,0] = 1
        self.wolf_2_pos = (1, 0)
        self.grid[1, 0] = 1
        self.wolf_3_pos = (2, 0)
        self.grid[2, 0] = 1


        #Try to place the sheep at the bottom middle of grid
        if self.grid_size % 2 == 0:
            self.sheep_pos = (int(self.grid_size/2), int(self.grid_size -1))
            if (self.grid[self.sheep_pos[0],self.sheep_pos[1]] == 1):
                raise ValueError('Wolf already placed there, grid probably too small')
            else:
                self.grid[self.sheep_pos[0],self.sheep_pos[1]] = 2
        else:
            self.sheep_pos = (int((self.grid_size+1)/2), int(self.grid_size -1))
            if (self.grid[self.sheep_pos[0],self.sheep_pos[1]] == 1):
                raise ValueError('Wolf already placed there, grid probably too small')
            else:
                self.grid[self.sheep_pos[0],self.sheep_pos[1]] = 2

        self.wolf_speed = wspeed
        self.sheep_speed = sspeed


    def move_wolf(self, wolf, dir):
        reward = 0

        if wolf == 0:
            current_pos = self.wolf_1_pos
        if wolf == 1:
            current_pos = self.wolf_2_pos
        if wolf == 2:
            current_pos = self.wolf_3_pos

        # If action is do nothing, just return
        if dir == 0:
            return 0

        # Action - move west
        if dir == 1:
            # If we dont go out of bounds and the new position is not another wolf, we can move the wolf
            if current_pos[1] - self.wolf_speed >= 0:
                return reward
            if self.grid[current_pos[0],current_pos[1] - self.wolf_speed] != 1:
                return reward
            new_pos = (current_pos[0], current_pos[1] - self.wolf_speed)
            if new_pos == self.sheep_pos:
                reward = 1

        # Action - move south
        if dir == 2:
            if current_pos[0] + self.wolf_speed < self.grid_size:
                return reward
            if self.grid[current_pos[0] + self.wolf_speed,current_pos[1]] != 1:
                return reward
            new_pos = (current_pos[0] + self.wolf_speed, current_pos[1])
            if new_pos == self.sheep_pos:
                reward = 1

        # Action - move east
        if dir == 3:
            if current_pos[1] - self.wolf_speed < self.grid_size:
                return reward
            if self.grid[current_pos[0],current_pos[1] - self.wolf_speed] != 1:
                return reward
            new_pos = (current_pos[0], current_pos[1] - self.wolf_speed)

            if new_pos == self.sheep_pos:
                reward = 1


        # Action - move north
        if dir == 4:
            if current_pos[0] - self.wolf_speed >= 0:
                return reward
            if self.grid[current_pos[0] - self.wolf_speed,current_pos[1]] != 1:
                return reward
            new_pos = (current_pos[0] - self.wolf_speed, current_pos[1])
            if new_pos == self.sheep_pos:
                reward = 1

        # Update positions
        #Set a wolf at the new position
        self.grid[new_pos[0],new_pos[1]] = 1
        #Set the old position as free
        self.grid[current_pos[0],current_pos[1]] = 0
        if wolf == 0:
            self.wolf_1_pos = new_pos
        if wolf == 1:
            self.wolf_2_pos = new_pos
        if wolf == 2:
            self.wolf_3_pos = new_pos
        return reward


    def move_sheep(self):
        #Moving the sheep randomly for now
        dir = random.randint(1,4)
        current_pos = self.sheep_pos

        # Action - move north
        if dir == 1:
            # If we dont go out of bounds and new position is free we can move the sheep
            if current_pos[1] - self.sheep_speed >= 0:
                return
            if self.grid[current_pos[0], (current_pos[1] - self.sheep_speed)] == 0:
               return
            else:
                new_pos = (current_pos[0], current_pos[1] - self.sheep_speed)

        # Action - move east
        if dir == 2:
            if current_pos[0] + self.sheep_speed < self.grid_size:
                return
            if self.grid[current_pos[0] + self.sheep_speed, current_pos[1]] == 0:
                return
            else:
                new_pos = (current_pos[0] + self.sheep_speed, current_pos[1])

        # Action - move south
        if dir == 3:
            if ((current_pos[1] + self.sheep_speed) >= self.grid_size):
                return
            if (self.grid[current_pos[0],current_pos[1] + self.sheep_speed] != 0):
                return
            else:
                new_pos = (current_pos[0], current_pos[1] + self.sheep_speed)

        # Action - move west
        if dir == 4:
            if current_pos[0] - self.sheep_speed >= 0:
                return
            if self.grid[current_pos[0] - self.sheep_speed,current_pos[1]] == 0:
                return
            else:
                new_pos = (current_pos[0] - self.sheep_speed, current_pos[1])

        # Update positions
        # Set a sheep at the new position
        self.grid[new_pos[0],new_pos[1]] = 2
        # Set the old position as free
        self.grid[current_pos[0],current_pos[1]] = 0
        self.sheep_pos = new_pos


    def frame_step(self, input_actions_tensor):
        reward = 0

        #Move all wolves
        # Actions meanings
        # action[0] == 1 -  wolf 1 do nothing
        # action[1] == 1 -  wolf 1  move north
        # action[2] == 1 -  wolf 1  move east
        # action[3] == 1 -  wolf 1  move south
        # action[4] == 1 -  wolf 1  move west
        # action[5] == 1 -  wolf 2 do nothing
        # etc etc up to
        # action[14] == 1 - wolf 3 move west
        input_actions = input_actions_tensor.cpu().numpy().astype(int)

        if sum(input_actions) != 1:
            raise ValueError('Not one action per wolf')

        wolf = 0
        dir = 0
        counter = 0
        for action in input_actions:
            if action == 1:
                reward = max(reward, State.move_wolf(self, wolf, dir))
            counter += 1
            if counter % 5 == 0:
                wolf += 1
            dir = counter % 5

        #Move the sheep
        State.move_sheep(self)

        return self.grid, reward