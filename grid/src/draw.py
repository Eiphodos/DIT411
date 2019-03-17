from graphics import *
import time

class Draw:
    def __init__(self, gridsize, grid, noDelay):
        self.grid_size = gridsize
        self.window = GraphWin("Wolfs and sheep", 150 * self.grid_size , 150 * self.grid_size)
        self.window.setBackground("lawn green")
        self.initialize_window(grid)
        if (noDelay):
            self.delay = 0
        else:
            self.delay = 0.01

    def initialize_window(self, grid):
        column_index  = 0
        for column in grid:
            row_index = 0
            for position in column:
                if position in (1,2,3):
                    self.draw_wolf(row_index, column_index, position)
                elif position == 4:
                    self.draw_sheep(row_index, column_index)
                row_index += 1
            column_index += 1

    def draw_wolf(self, row, column, wolf): 
        if wolf == 1:
            self.wolf1_pos = 150*row + 75, 150*column + 75
            cp = Point(self.wolf1_pos[0], self.wolf1_pos[1])
            self.wolf1 = Circle(cp, 25)
            self.wolf1.setFill("brown4")
            self.wolf1.draw(self.window)
        if wolf == 2:
            self.wolf2_pos = 150*row + 75, 150*column + 75
            cp = Point(self.wolf2_pos[0], self.wolf2_pos[1])
            self.wolf2 = Circle(cp, 25)
            self.wolf2.setFill("brown4")
            self.wolf2.draw(self.window)
        if wolf == 3:
            self.wolf3_pos = 150*row + 75, 150*column + 75
            cp = Point(self.wolf3_pos[0], self.wolf3_pos[1])
            self.wolf3 = Circle(cp, 25)
            self.wolf3.setFill("brown4")
            self.wolf3.draw(self.window)

    def draw_sheep(self, row, column):
        self.sheep_pos = 150*row + 75, 150*column + 75
        cp = Point(self.sheep_pos[0], self.sheep_pos[1])
        self.sheep = Circle(cp, 25)
        self.sheep.setFill("snow")
        self.sheep.draw(self.window)

    def update_window(self, grid):
        
        column_index = 0
        for row in grid:
            row_index  = 0
            for position in row:
                if position in (1,2,3):
                    self.move_wolf(row_index, column_index, position)
                elif position == 4:
                    self.move_sheep(row_index, column_index)
                time.sleep(self.delay)
                row_index += 1
            column_index += 1

    def move_wolf(self, row, column, wolf):
        if wolf == 1:
            new_pos = 150*row + 75, 150*column + 75
            move_x = new_pos[0] - self.wolf1_pos[0]
            move_y = new_pos[1] - self.wolf1_pos[1]
            self.wolf1.move(move_x, move_y)
            self.wolf1_pos = new_pos
        if wolf == 2:
            new_pos = 150*row + 75, 150*column + 75
            move_x = new_pos[0] - self.wolf2_pos[0]
            move_y = new_pos[1] - self.wolf2_pos[1]
            self.wolf2.move(move_x, move_y)
            self.wolf2_pos = new_pos
        if wolf == 3:
            new_pos = 150*row + 75, 150*column + 75
            move_x = new_pos[0] - self.wolf3_pos[0]
            move_y = new_pos[1] - self.wolf3_pos[1]
            self.wolf3.move(move_x, move_y)
            self.wolf3_pos = new_pos

    def move_sheep(self, row, column):
        new_pos = 150*row + 75, 150*column + 75
        move_x = new_pos[0] - self.sheep_pos[0]
        move_y = new_pos[1] - self.sheep_pos[1]
        self.sheep.move(move_x, move_y)
        self.sheep_pos = new_pos
