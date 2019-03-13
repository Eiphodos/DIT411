import gym
import math
import random


class NeuralNetwork():


    def left(self):
        print("hej")

    def right(self):
        print("hej")

    def up(self):
        print("hej")

    def down(self):
        print("hej")



class Entity:
    def __init__(self, animal, posX, posY, rot, numberOfDSight, length, width, index):
        self.posX = posX
        self.posY = posY
        self.rot = rot
        self.eyes = numberOfDSight
        self.length = length
        self.width = width
        self.internalRandom = random.seed()
        self.speed = 0
        self.index = index
        if animal == "Wolf":
            self.acceleration = 1
            self.deceleration = 0.5
            self.maxspeed = 2
            self.minspeed = -1
            self.animal = 1
            self.rotationSpeed = 1.0/6
        if animal == "Sheep":
            self.acceleration = 1
            self.deceleration = 0.5
            self.maxspeed = 2
            self.minspeed = -1
            self.animal = 1
            self.rotationSpeed = 1.0 / 6

    def move(self):
        rotTemp = self.rot%360
        speedX = math.cos(rotTemp * math.pi/180)*self.speed
        speedY = math.sin(rotTemp * math.pi/180)*self.speed
        self.posX += speedX
        self.posY += speedY

    def inputChange(self):
        #a = NeuralNetwork()
        #a.funca()
        if(self.animal == 1):
            self.speedChange((random.randrange(-10, 10)/10.0))
            self.rotationChange(random.randrange(-3600, 3600)/10.0)


    def speedChange(self, speedChange):
        if(self.speed + speedChange < 0):
            if(self.speed + speedChange * self.deceleration < self.minspeed):
                self.speed = self.minspeed
            else:
                self.speed += speedChange * self.deceleration
        else:
            if((self.speed + speedChange * self.acceleration) > self.maxspeed):
                self.speed = self.maxspeed
            else:
                self.speed += speedChange * self.acceleration

    def rotationChange(self, rotationChange):
        self.rot = rotationChange*self.rotationSpeed

    def asString(self):
        tempString = ""
        tempString += str(self.index) + "|"
        #tempString += str(self.animal) + "|"
        tempString += str(round(self.posX, 3)) + "," + str(round(self.posY, 3)) + "|"
        tempString += str(round(self.rot, 3)) + "|"
        #tempString += str(round(self.length, 3)) + "," + str(round(self.width, 3))
        return tempString + "O"

    def objectInf(self):
        tempString = ""
        tempString += str(self.index) + "|"
        tempString += str(self.animal) + "|"
        tempString += str(round(self.length, 3)) + "," + str(round(self.width, 3))
        return tempString + "O"

p1 = Entity("Wolf", 0, 400, 0, 270, 20, 10, 1)
#p2 = Entity("Wolf", 0, 300, 0, 270, 20, 10, 2)
#p3 = Entity("Sheep", 500, 700, 0, 81, 20, 10, 2)

saveFile = ""

for y in range(0, 10):
    for x in range(0, 10000):
        p1.inputChange()
        p1.move()
#        p2.inputChange()
 #       p2.move()
        saveFile += "T" + p1.asString() #+ p2.asString()
    saveFile += "\n"

endFile = ""

endFile = p1.objectInf() #+ p2.objectInf() + "B"

endFile += saveFile


with open("Output.txt", "w") as text_file:
    text_file.write(endFile)




