import numpy as np
import random

class State:
    def __init__(self, gridsize, wspeed, sspeed):
        #Free position = 0, wolf = 1, sheep = 2
        self.grid_size = gridsize
        self.grid = np.zeros((gridsize, gridsize), dtype=np.int)
        self.n_wolves = 3
        self.finished = False

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
            if current_pos[1] - self.wolf_speed < 0:
                return reward
            if self.grid[current_pos[0],current_pos[1] - self.wolf_speed] == 1:
                return reward
            new_pos = (current_pos[0], current_pos[1] - self.wolf_speed)
            if new_pos == self.sheep_pos:
                reward = 1
                self.finished = True

        # Action - move south
        if dir == 2:
            if current_pos[0] + self.wolf_speed >= self.grid_size:
                return reward
            if self.grid[current_pos[0] + self.wolf_speed,current_pos[1]] == 1:
                return reward
            new_pos = (current_pos[0] + self.wolf_speed, current_pos[1])
            if new_pos == self.sheep_pos:
                reward = 1
                self.finished = True

        # Action - move east
        if dir == 3:
            if current_pos[1] + self.wolf_speed >= self.grid_size:
                return reward
            if self.grid[current_pos[0],current_pos[1] + self.wolf_speed] == 1:
                return reward
            new_pos = (current_pos[0], current_pos[1] + self.wolf_speed)
            if new_pos == self.sheep_pos:
                reward = 1
                self.finished = True


        # Action - move north
        if dir == 4:
            if current_pos[0] - self.wolf_speed < 0:
                return reward
            if self.grid[current_pos[0] - self.wolf_speed,current_pos[1]] == 1:
                return reward
            new_pos = (current_pos[0] - self.wolf_speed, current_pos[1])
            if new_pos == self.sheep_pos:
                reward = 1
                self.finished = True
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

        print(self.grid)
        print(direction_preference)
        print("Sheep: " + str(self.sheep_pos))
        print("Closest wolf: " + str(closest_wolf_pos))
        print(self.wolf_1_pos)
        print(self.wolf_2_pos)
        print(self.wolf_3_pos)

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

        if (reward == 1):
            #Sheep was caught so return
            return self.grid, reward, self.finished
        else:
            #Move the sheep
            State.move_sheep(self)
            return self.grid, reward, self.finished