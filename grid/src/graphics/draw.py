from graphics import *

class Draw:
    def __init__(self, gridsize):
        self.grid_size = gridsize
        self.window = GraphWin("Wolfs and sheep", 50 * self.grid_size , 50 * self.grid_size)
        self.window.setBackground("lawn green")

    def initialize_window(self, grid):
        row_index  = 0
        column_index = 0
        for row in grid:
            for position in row:
                if position in (1,2,3):
                    draw_wolf(row_index, column_index, position)
                elif position == 4:
                    draw_sheep(row_index, column_index)
                column_index += 1
            row_index += 1

    def draw_wolf(self, row, column, wolf):
        cp = Point(50*row + 25, 50*column + 25)
        if position == 1:
            self.wolf1 = Circle(cp, 15)
            self.wolf1.setFill("brown4")
            self.wolf1.draw(self.window)
        if position == 2:
            self.wolf2 = Circle(cp, 15)
            self.wolf2.setFill("brown4")
            self.wolf2.draw(self.window)
        if position == 3:
            self.wolf3 = Circle(cp, 15)
            self.wolf3.setFill("brown4")
            self.wolf3.draw(self.window)

    def draw_sheep(self, row, column):
        cp = Point(50*row + 25, 50*column + 25)
        self.sheep = Circle(cp, 15)
        self.sheep.setFill("snow")
        self.sheep.draw(self.window)

    def update_window(self, grid):
        row_index  = 0
        column_index = 0
        for row in grid:
            for position in row:
                if position in (1,2,3):
                    move_wolf(row_index, column_index, position)
                elif position == 4:
                    move_sheep(row_index, column_index)
                column_index += 1
            row_index += 1

    def move_wolf(self, row, column, wolf):
        if position == 1:
            self.wolf1.move(50*row + 25, 50*column + 25)
        if position == 2:
            self.wolf2.move(50*row + 25, 50*column + 25)
        if position == 3:
            self.wolf3.move(50*row + 25, 50*column + 25)

    def move_sheep(self, row, column):
        self.sheep.move(50*row + 25, 50*column + 25)