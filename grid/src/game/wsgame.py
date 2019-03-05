import numpy as np
import random

class State:
    def __init__(self, gridsize, wspeed, sspeed):
        #Free position = 0, wolf1 = 1, wolf2 = 2, wolf3 = 3, sheep = 4
        self.grid_size = gridsize
        self.grid = np.zeros((gridsize, gridsize), dtype=np.int)
        self.n_wolves = 3

        if (self.grid_size * self.grid_size < 9 ):
            raise ValueError('The grid is too small!')

        self.wolf_1_pos = (0, 0)
        self.grid[0,0] = 1
        self.wolf_2_pos = (1, 0)
        self.grid[1, 0] = 2
        self.wolf_3_pos = (2, 0)
        self.grid[2, 0] = 3


        #Try to place the sheep at the bottom middle of grid
        if self.grid_size % 2 == 0:
            self.sheep_pos = (int(self.grid_size/2), int(self.grid_size -1))
            if (self.grid[self.sheep_pos[0],self.sheep_pos[1]] in (1, 2, 3)):
                raise ValueError('Wolf already placed there, grid probably too small')
            else:
                self.grid[self.sheep_pos[0],self.sheep_pos[1]] = 4
        else:
            self.sheep_pos = (int((self.grid_size+1)/2), int(self.grid_size -1))
            if (self.grid[self.sheep_pos[0],self.sheep_pos[1]] in (1, 2, 3)):
                raise ValueError('Wolf already placed there, grid probably too small')
            else:
                self.grid[self.sheep_pos[0],self.sheep_pos[1]] = 4

        self.wolf_speed = wspeed
        self.sheep_speed = sspeed


    def move_wolf(self, wolf, dir):
        reward = 0
        # Reward (punishment) for going out of bounds
        reward_oob = -0.1
        # Reward (punishment) for trying to enter an already occupied position
        reward_ao = -0.01
        # Reward for catching the sheep
        reward_catch = 1

        if wolf == 0:
            current_pos = self.wolf_1_pos
        if wolf == 1:
            current_pos = self.wolf_2_pos
        if wolf == 2:
            current_pos = self.wolf_3_pos

        
        # If action is do nothing, just return
        if dir == 0:
            return reward
        # Action - move west
        if dir == 1:
            # Check for out of bounds
            if current_pos[1] - self.wolf_speed < 0:
                reward = reward_oob
                return reward
            # Check if another wolf is already there
            if self.grid[current_pos[0],current_pos[1] - self.wolf_speed] in (1, 2, 3):
                reward = reward_ao
                return reward
            new_pos = (current_pos[0], current_pos[1] - self.wolf_speed)
            if new_pos == self.sheep_pos:
                reward = reward_catch

        # Action - move south
        if dir == 2:
            if current_pos[0] + self.wolf_speed >= self.grid_size:
                reward = reward_oob
                return reward
            if self.grid[current_pos[0] + self.wolf_speed,current_pos[1]] in (1, 2, 3):
                reward = reward_ao
                return reward
            new_pos = (current_pos[0] + self.wolf_speed, current_pos[1])
            if new_pos == self.sheep_pos:
                reward = reward_catch

        # Action - move east
        if dir == 3:
            if current_pos[1] + self.wolf_speed >= self.grid_size:
                reward = reward_oob
                return reward
            if self.grid[current_pos[0],current_pos[1] + self.wolf_speed] in (1, 2, 3):
                reward = reward_ao
                return reward
            new_pos = (current_pos[0], current_pos[1] + self.wolf_speed)
            if new_pos == self.sheep_pos:
                reward = reward_catch


        # Action - move north
        if dir == 4:
            if current_pos[0] - self.wolf_speed < 0:
                reward = reward_oob
                return reward
            if self.grid[current_pos[0] - self.wolf_speed,current_pos[1]] in (1, 2, 3):
                reward = reward_ao
                return reward
            new_pos = (current_pos[0] - self.wolf_speed, current_pos[1])
            if new_pos == self.sheep_pos:
                reward = reward_catch

        # Update positions
        #Set the old position as free
        self.grid[current_pos[0],current_pos[1]] = 0
        #Set a wolf at the new position
        if wolf == 0:
            self.grid[new_pos[0],new_pos[1]] = 1
            self.wolf_1_pos = new_pos
        if wolf == 1:
            self.grid[new_pos[0],new_pos[1]] = 2
            self.wolf_2_pos = new_pos
        if wolf == 2:
            self.grid[new_pos[0],new_pos[1]] = 3
            self.wolf_3_pos = new_pos
        return reward


    def move_sheep(self):

        current_pos = self.sheep_pos

        # Test available directions
        north_avail = False
        south_avail = False
        west_avail = False
        east_avail = False

        # Testing if west is available
        if (current_pos[1] - self.sheep_speed >= 0 and self.grid[current_pos[0], (current_pos[1] - self.sheep_speed)] == 0):
            west_avail = True
        # Testing if east is available
        if (current_pos[1] + self.sheep_speed < self.grid_size and self.grid[current_pos[0], (current_pos[1] + self.sheep_speed)] == 0):
            east_avail = True
        # Testing if south is available
        if (current_pos[0] + self.sheep_speed < self.grid_size and self.grid[(current_pos[0] + self.sheep_speed), current_pos[1] ] == 0):
            south_avail = True
        # Testing if north is available
        if (current_pos[0] - self.sheep_speed >= 0 and self.grid[(current_pos[0] - self.sheep_speed), current_pos[1] ] == 0):
            north_avail = True

        if (not north_avail and not south_avail and not west_avail and not east_avail):
            return

        estimated_dist_w1 = abs(current_pos[0] - self.wolf_1_pos[0])  + abs(current_pos[1] - self.wolf_1_pos[1])
        estimated_dist_w2 = abs(current_pos[0] - self.wolf_2_pos[0])  + abs(current_pos[1] - self.wolf_2_pos[1])
        estimated_dist_w3 = abs(current_pos[0] - self.wolf_3_pos[0])  + abs(current_pos[1] - self.wolf_3_pos[1])

        if (estimated_dist_w1 < estimated_dist_w2):
            if (estimated_dist_w1 < estimated_dist_w3):
                closest_wolf_pos = self.wolf_1_pos
            else:
                closest_wolf_pos = self.wolf_3_pos
        else:
            if (estimated_dist_w2 < estimated_dist_w3):
                closest_wolf_pos = self.wolf_2_pos
            else:
                closest_wolf_pos = self.wolf_3_pos

        # Create a list of prefered directions to move to based on the closest wolfs position

        direction_preference = [0,0,0,0]

        # Check if the closest wolf is closer in the Y direction compared to the X direction
        if (abs(current_pos[0] - closest_wolf_pos[0]) < abs(current_pos[1] - closest_wolf_pos[1])):
            # If it is, check if the wolf is north or south of the wolf and decide what way to flee based on that.
            if (current_pos[0] < closest_wolf_pos[0]):
                # Flee north as highest prio
                direction_preference[0] = 4
                # Flee south as lowest prio
                direction_preference[3] = 2
            else:
                # Flee south as highest prio
                direction_preference[0] = 2
                # Flee north as lowest prio
                direction_preference[3] = 4
            if (current_pos[1] < closest_wolf_pos[1]):
                # Flee west as second highest prio
                direction_preference[1] = 1
                # Flee east as second to last prio
                direction_preference[2] = 3
            else:
                # Flee east as second highest prio
                direction_preference[1] = 3
                # Flee west as second to last prio
                direction_preference[2] = 1
        else:
        # Else if wolf is closer in the X direction we update the priority list based on that instead
            if (current_pos[1] < closest_wolf_pos[1]):
                # Flee west as highest prio
                direction_preference[0] = 1
                # Flee east as last prio
                direction_preference[3] = 3
            else:
                # Flee east as highest prio
                direction_preference[0] = 3
                # Flee west as second to last prio
                direction_preference[3] = 1
            if (current_pos[0] < closest_wolf_pos[0]):
                # Flee north as second highest prio
                direction_preference[1] = 4
                # Flee south as second to last prio
                direction_preference[2] = 2
            else:
                # Flee south as second highest prio
                direction_preference[1] = 2
                # Flee north as second to last prio
                direction_preference[2] = 4


        # Set sheeps new position to the highest priority one
        successful_escape = False
        i = 0
        while (not successful_escape):
            if (direction_preference[i] == 1 and west_avail):
                new_pos = (current_pos[0], current_pos[1] - self.sheep_speed)
                successful_escape = True
            elif (direction_preference[i] == 2 and south_avail):
                new_pos = (current_pos[0] + self.sheep_speed, current_pos[1])
                successful_escape = True
            elif (direction_preference[i] == 3 and east_avail):
                new_pos = (current_pos[0], current_pos[1] + self.sheep_speed)
                successful_escape = True
            elif (direction_preference[i] == 4 and north_avail):
                new_pos = (current_pos[0] - self.sheep_speed, current_pos[1])
                successful_escape = True
            i += 1

        # Update positions
        # Set a sheep at the new position
        self.grid[new_pos[0],new_pos[1]] = 4
        # Set the old position as free
        self.grid[current_pos[0],current_pos[1]] = 0
        self.sheep_pos = new_pos


    def frame_step(self, tensor_action1, tensor_action2, tensor_action3):
        reward1 = 0
        reward2 = 0
        reward3 = 0

        #Move wolves
        # Actions meanings
        # action[0] == 1 -  wolf do nothing
        # action[1] == 1 -  wolf move north
        # action[2] == 1 -  wolf move east
        # action[3] == 1 -  wolf move south
        # action[4] == 1 -  wolf move west
        tensor_action1 = tensor_action1.cpu().numpy().astype(int)
        tensor_action2 = tensor_action2.cpu().numpy().astype(int)
        tensor_action3 = tensor_action3.cpu().numpy().astype(int)

        if sum(tensor_action1) != 1 or sum(tensor_action2) != 1 or sum(tensor_action3) != 1:
            raise ValueError('Not one action per wolf')

        wolf = 0
        dir = 0
        counter = 0
        for action in tensor_action1:
            if action == 1:
                reward1 = self.move_wolf(0, dir)
            counter += 1
            dir = counter % 5

        for action in tensor_action2:
            if action == 1:
                reward2 = self.move_wolf(1, dir)
            counter += 1
            dir = counter % 5

        for action in tensor_action3:
            if action == 1:
                reward3 = self.move_wolf(2, dir)
            counter += 1
            dir = counter % 5
        print(self.grid)
        if (reward1 == 1 or reward2 == 1 or reward3 == 1 ):
            #Sheep was caught so set reward to 1 for all wolves and reset game state
            reward1 = 1
            reward2 = 1
            reward3 = 1
            self.__init__(self.grid_size, self.wolf_speed, self.sheep_speed)
            return self.grid, reward1, reward2, reward3, True
        else:
            #Move the sheep
            self.move_sheep()
            return self.grid, reward1, reward2, reward3, False