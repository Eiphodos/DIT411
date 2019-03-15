import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

global nextReady;

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):

    def basic_policy(self):
        print("action")
        # take action


class Entity:
    def __init__(self, animal, posX, posY, rot, numberOfDSight, length, width, index, game):
        self.posX = posX
        self.game = game
        self.posY = posY
        self.rot = rot
        self.eyes = numberOfDSight
        self.length = length
        self.width = width
        self.internalRandom = random.seed()
        self.speed = 0
        self.index = index
        self.reward = 0
        self.done = False
        if animal == "Wolf":
            self.acceleration = 1
            self.deceleration = 0.5
            self.maxspeed = 2
            self.minspeed = -1
            self.animal = 1
            self.rotationSpeed = 1.0 / 6
        if animal == "Sheep":
            self.acceleration = 1
            self.deceleration = 0.5
            self.maxspeed = 2
            self.minspeed = -1
            self.animal = 2
            self.rotationSpeed = 1.0 / 6

    def move(self):
        rotTemp = self.rot % 360
        speedX = math.cos(rotTemp * math.pi / 180) * self.speed
        speedY = math.sin(rotTemp * math.pi / 180) * self.speed

        self.posX += speedX
        self.posY += speedY

        self.posX = max(self.posX, self.game.leftLimit)
        self.posX = min(self.posX, self.game.rightLimit)

        self.posY = max(self.posY, self.game.bottomLimit)
        self.posY = min(self.posY, self.game.topLimit)

    def inputChange(self, actions):
        # a = NeuralNetwork()
        # a.funca()
        if (self.animal == 1):
            self.speedChange(actions[0])
            self.rotationChange(10 * actions[1])

            vision = self.game.getWolfVision(self.index)
            vision.append(self.speed / self.maxspeed)

            for i in range(vision):
                self.game.state.append[vision[i]]


            self.game.wolfVision.append(vision);

            if (self.game.debug):
                print("x: " + str(self.posX))
                print("y: " + str(self.posY))
                print("r: " + str(self.rot))

            if (self.game.debug):
                for i in range(81):
                    value = vision[i]
                    if (value != 0):
                        input = "unknown"
                        if (i < 27):
                            input = "sheep: "
                        elif (i < 54):
                            input = "wolf: "
                        else:
                            input = "Wall: "
                        print(input + str(vision[i]))

                print("_______________________________________________")

        elif (self.animal == 2):
            self.speedChange((random.randrange(-10, 10) / 10.0))
            self.rotationChange(random.randrange(-3600, 3600) / 10.0)

            self.game.sheepPosition[0] = self.posX
            self.game.sheepPosition[1] = self.posY

    def speedChange(self, speedChange):
        if (self.speed + speedChange < 0):
            if (self.speed + speedChange * self.deceleration < self.minspeed):
                self.speed = self.minspeed
            else:
                self.speed += speedChange * self.deceleration
        else:
            if ((self.speed + speedChange * self.acceleration) > self.maxspeed):
                self.speed = self.maxspeed
            else:
                self.speed += speedChange * self.acceleration

    def rotationChange(self, rotationChange):
        self.rot = rotationChange * self.rotationSpeed

    def asString(self):
        tempString = ""
        tempString += str(self.index) + "|"
        # tempString += str(self.animal) + "|"
        tempString += str(round(self.posX, 3)) + "," + str(round(self.posY, 3)) + "|"
        tempString += str(round(self.rot, 3)) + "|"
        # tempString += str(round(self.length, 3)) + "," + str(round(self.width, 3))
        return tempString + "O"

    def getCurrentState(self):
        cState = self.game.getWolfVision(self.index)
        cState.append(self.speed / self.maxspeed)
        cState.append((self.rot % 360) / 360)

    def getNextState(self):
        if (nextReady == 1):
            return self.getCurrentState()

    def done(self):
        return self.done

    def objectInf(self):
        tempString = ""
        tempString += str(self.index) + "|"
        tempString += str(self.animal) + "|"
        tempString += str(round(self.length, 3)) + "," + str(round(self.width, 3))
        return tempString + "O"


class Game:
    def __init__(self):
        self.index = 0
        self.gen = 0
        self.saveFile = ""
        self.animals = []
        self.radius = 30
        self.sheepPosition = [240.0, 175.0]
        self.visionLength = 100
        self.fieldOfVision = 270
        self.nVisionLines = 27
        self.wolfPosition = (350, 200)
        self.wolfRotation = -30
        # set to 0 for no print 1 for print
        self.debug = 0

        # limits
        self.bottomLimit = 0
        self.topLimit = 1000
        self.leftLimit = 0
        self.rightLimit = 1000
        self.wallLines = [
            [(self.leftLimit, self.bottomLimit), (self.leftLimit, self.topLimit)],
            [(self.rightLimit, self.bottomLimit), (self.rightLimit, self.topLimit)],
            [(self.leftLimit, self.bottomLimit), (self.rightLimit, self.bottomLimit)],
            [(self.leftLimit, self.topLimit), (self.rightLimit, self.topLimit)]
        ]

        lines = self.createLines(self.nVisionLines, self.visionLength, self.fieldOfVision, self.wolfRotation,
                                 self.wolfPosition)
        wallVisionLines = self.createLines(self.nVisionLines, self.visionLength, self.fieldOfVision, self.wolfRotation,
                                           self.wolfPosition)

        self.animals = []

        self.animals.append(Entity("Sheep", 50, 50, 0, 270, 15, 10, 1, self))
        self.animals.append(Entity("Wolf", 100, 0, 0, 270, 20, 10, 2, self))
        self.animals.append(Entity("Wolf", 0, 100, 0, 270, 20, 10, 3, self))
        self.animals.append(Entity("Wolf", 350, 350, 0, 270, 20, 10, 4, self))

        self.state = []


    def gameReset(self):
        self.gen = self.gen + 1
        self.animals = []

        self.animals.append(Entity("Sheep", 50, 50, 0, 270, 15, 10, 1, self))
        self.animals.append(Entity("Wolf", 100, 0, 0, 270, 20, 10, 2, self))
        self.animals.append(Entity("Wolf", 0, 100, 0, 270, 20, 10, 3, self))
        self.animals.append(Entity("Wolf", 350, 350, 0, 270, 20, 10, 4, self))
        self.saveFile += "\n"

    def controllableAgentAmount(self):
        return len(self.animals) - 1

    def nextState(self):
        self.state = []
        print("Action: " + str(self.index))
        self.index = self.index + 1
        self.saveFile += "T"
        for i in range(len(self.animals)):
            #self.animals[i].inputChange(actionArray[i]);
            self.animals[i].move();
            self.saveFile += self.animals[i].asString()


    def conductAction(self, actionArray):
        #for i in range(len(self.animals)):
            #self.animals[i].inputChange(actionArray[i]);


        return self.getCurrentState(), self.getReward(), self.done(), 1

    def done(self):
        endFile = ""

        for i in range(len(self.animals)):

            #these two should be removed?
            #self.animals[i].inputChange();
            #self.animals[i].move();

            self.saveFile += self.animals[i].asString()
            endFile += self.animals[i].objectInf()

        endFile += "B"
        endFile += self.saveFile

        with open("Output.txt", "w") as text_file:
            text_file.write(endFile)

        if(self.index > 500):
            return True
        else:
            return False

    def getCurrentState(self):
        self.state = []
        for i in range(len(self.animals)):
            self.state += self.getWolfVision(self.animals[i].index)
        return self.state

    def getReward(self):
        reward = 0
        closestSheepDistance = 100000
        for i in range(len(self.animals)):
            animal = self.animals[i]

            #is wolf
            if(animal.animal == 1):
                xDelta = self.sheepPosition[0] - animal.posX
                yDelta = self.sheepPosition[1] - animal.posY

                distanceToSheep = math.sqrt(xDelta**2 +yDelta**2)

                closestSheepDistance = min(closestSheepDistance, distanceToSheep)

            if (closestSheepDistance < self.radius):
                reward = 1000
            else:
                reward = 1000 - closestSheepDistance;
        return reward;

    def getWolfVision(self, wolfIndexPassed):

        wolf = self.animals[wolfIndexPassed - 1];
        currentPosition = (wolf.posX, wolf.posY)

        lines = self.createLines(self.nVisionLines, self.visionLength, self.fieldOfVision, wolf.rot, currentPosition)

        wallVisionLines = self.createLines(self.nVisionLines, self.visionLength, self.fieldOfVision, wolf.rot,
                                           currentPosition)

        wolfVision = []

        # TODO multiple sheep?   - like wolves but animals[i].animal == 2

        # lines towards sheep
        for lineIndex in range(self.nVisionLines):
            dist = self.distanceToObject(self.radius, self.sheepPosition, lines[lineIndex][0], lines[lineIndex][1],
                                         self.visionLength);

            partOFMaxDistance = dist / float(self.visionLength + self.radius);
            partOFMaxDistance = max(0, partOFMaxDistance)
            partOFMaxDistance = min(1, partOFMaxDistance)

            if (partOFMaxDistance > 0):
                partOFMaxDistance = 1 - partOFMaxDistance;

            wolfVision.append(partOFMaxDistance)

        # lines towards wolf
        for lineIndex in range(self.nVisionLines):

            maxPartOfMaxDistance = 0;

            for wolfIndex in range(len(self.animals)):

                # not self and is wolf
                if (wolfIndex != wolfIndexPassed and self.animals[wolfIndex].animal == 1):

                    wolfPosition = (self.animals[wolfIndex].posX, self.animals[wolfIndex].posY)

                    xDelta = currentPosition[0] - wolfPosition[0]
                    yDelta = currentPosition[1] - wolfPosition[1]

                    distance = math.sqrt(xDelta ** 2 + yDelta ** 2)
                    if (distance < self.visionLength + self.radius):
                        dist = self.distanceToObject(self.radius, wolfPosition, lines[lineIndex][0],
                                                     lines[lineIndex][1],
                                                     self.visionLength);

                        partOFMaxDistance = dist / float(self.visionLength + self.radius);
                        partOFMaxDistance = max(0, partOFMaxDistance)
                        partOFMaxDistance = min(1, partOFMaxDistance)

                        if (partOFMaxDistance > 0):
                            partOFMaxDistance = 1 - partOFMaxDistance;

                        maxPartOfMaxDistance = max(maxPartOfMaxDistance, partOFMaxDistance)

            wolfVision.append(maxPartOfMaxDistance)

        # lines towards walls
        for lineIndex in range(self.nVisionLines):
            dist = self.distanceFromWall(wallVisionLines[lineIndex]);

            if (dist < 0 or dist > self.visionLength):
                wolfVision.append(0)
            else:
                # hit and is < visionlength from orgin

                partOFMaxDistance = dist / float(self.visionLength);
                partOFMaxDistance = min(1, partOFMaxDistance)
                partOFMaxDistance = max(0, partOFMaxDistance)

                if (partOFMaxDistance > 0):
                    partOFMaxDistance = 1 - partOFMaxDistance;

                wolfVision.append(partOFMaxDistance)

        return wolfVision;

    def distanceToObject(self, radius, objPos, line1, line2, visionLength):

        distToline1 = math.sqrt((line1[0] - objPos[0]) ** 2 + (line1[1] - objPos[1]) ** 2)
        distToline2 = math.sqrt((line2[0] - objPos[0]) ** 2 + (line2[1] - objPos[1]) ** 2)

        if (distToline1 > visionLength + radius or distToline2 > visionLength + radius):
            return -1;

        x = np.array(objPos)
        u = np.array(line1)
        v = np.array(line2)

        n = v - u

        # n /= np.linalg.norm(n, 2)
        n /= visionLength

        intersectPos = u + n * np.dot(x - u, n)

        distanceLineToObject = math.sqrt((intersectPos[0] - objPos[0]) ** 2 + (intersectPos[1] - objPos[1]) ** 2)

        if (distanceLineToObject <= radius):
            orginToIntersectDistance = math.sqrt((line1[0] - intersectPos[0]) ** 2 + (line1[1] - intersectPos[1]) ** 2)

            if (
                    orginToIntersectDistance < visionLength and distToline1 <= visionLength + radius and distToline2 <= visionLength + radius):
                return (orginToIntersectDistance)
            elif (distToline1 <= radius or distToline2 <= radius):
                return (orginToIntersectDistance)
            else:
                return (-1.0)

        else:
            return -1;

    def createLines(self, nLines, lineLength, fieldOfView, startAngle, orgin):
        startX = orgin[0]
        startY = orgin[1]

        degreesPerLine = fieldOfView / (nLines - 1)

        linesStartAtAngle = startAngle - fieldOfView / 2;

        lines = [0 for x in range(nLines)]

        for lineIndex in range(nLines):
            angle = linesStartAtAngle + lineIndex * degreesPerLine;

            stopX = startX + math.sin(math.radians(angle)) * lineLength
            stopY = startY + math.cos(math.radians(angle)) * lineLength

            lines[lineIndex] = [(startX, startY), (stopX, stopY)]

        return lines

    def line_intersection(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return "none"

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div

        return x, y

    def distanceFromWall(self, visionLine):
        closestDistance = 1000000000;

        for lineIndex in range(len(self.wallLines)):
            intersection = self.line_intersection((self.wallLines[lineIndex][0], self.wallLines[lineIndex][1]),
                                                  (visionLine[0], visionLine[1]))

            if (intersection == "none"):
                return -1;
            else:
                xDeltaEnd = visionLine[1][0] - intersection[0]
                yDeltaEnd = visionLine[1][1] - intersection[1]

                distanceEnd = math.sqrt(xDeltaEnd ** 2 + yDeltaEnd ** 2)

                if (distanceEnd <= self.visionLength):
                    xDelta = visionLine[0][0] - intersection[0]
                    yDelta = visionLine[0][1] - intersection[1]

                    distance = math.sqrt(xDelta ** 2 + yDelta ** 2)

                    closestDistance = min(closestDistance, distance)

        if (closestDistance == 1000000000):
            return -1;

        return closestDistance;


game = Game()
for y in range(0, 10):
    for x in range(0, 1000):
        pass#game.nextState()
    game.gameReset()

game.done()
