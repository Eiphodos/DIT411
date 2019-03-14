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
            #take action



def distanceToObject(radius, objPos, line1, line2, visionLength):

    distToline1 = math.sqrt( (line1[0] - objPos[0])**2 + (line1[1] - objPos[1])**2 )
    distToline2 = math.sqrt( (line2[0] - objPos[0])**2 + (line2[1] - objPos[1])**2 )

    if(distToline1 > visionLength + radius or distToline2 > visionLength + radius):
        return -1;

    x = np.array(objPos)
    u = np.array(line1)
    v = np.array(line2)

    n = v - u

    #n /= np.linalg.norm(n, 2)
    n /= visionLength

    intersectPos = u + n*np.dot(x - u, n)

    distanceLineToObject = math.sqrt( (intersectPos[0] - objPos[0])**2 + (intersectPos[1] - objPos[1])**2 )

    if(distanceLineToObject <= radius):
        orginToIntersectDistance = math.sqrt( (line1[0] - intersectPos[0])**2 + (line1[1] - intersectPos[1])**2 )

        if(orginToIntersectDistance < visionLength and distToline1 <= visionLength + radius and distToline2 <= visionLength + radius):
            return (orginToIntersectDistance)
        elif(distToline1 <= radius or distToline2 <= radius):
            return (orginToIntersectDistance)
        else:
            return (-1.0)

    else:
        return -1;

def createLines(nLines, lineLength, fieldOfView, startAngle, orgin):
    startX = orgin[0]
    startY = orgin[1]

    degreesPerLine = fieldOfView / (nLines-1)

    linesStartAtAngle = startAngle - fieldOfView/2;

    lines = [0 for x in range(nLines)]

    for lineIndex in range(nLines):

        angle = linesStartAtAngle + lineIndex * degreesPerLine;

        stopX = startX + math.sin( math.radians(angle) ) * lineLength
        stopY = startY + math.cos( math.radians(angle) ) * lineLength

        lines[lineIndex] = [(startX, startY),(stopX, stopY)]

    return lines

def line_intersection(line1, line2):
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

def distanceFromWall(visionLine):
    closestDistance = 1000000000;

    for lineIndex in range(len(wallLines)):
        intersection = line_intersection( (wallLines[lineIndex][0], wallLines[lineIndex][1]), (visionLine[0], visionLine[1]))

        if (intersection == "none"):
            return -1;
        else:
            xDeltaEnd = visionLine[1][0] - intersection[0]
            yDeltaEnd = visionLine[1][1] - intersection[1]

            distanceEnd = math.sqrt( xDeltaEnd**2 + yDeltaEnd**2 )

            if(distanceEnd <= visionLength):
                xDelta = visionLine[0][0] - intersection[0]
                yDelta = visionLine[0][1] - intersection[1]

                distance = math.sqrt( xDelta**2 + yDelta**2 )

                closestDistance = min(closestDistance, distance)

    if(closestDistance == 1000000000 ):
        return -1;

    return closestDistance;

radius = 30
sheepPosition = [240.0, 175.0]
visionLength = 100
fieldOfVision = 270
nVisionLines = 27
wolfPosition = (350, 200)
wolfRotation = -30

#set to 0 for no print 1 for print
debug = 0

#limits
bottomLimit = 0
topLimit = 1000
leftLimit = 0
rightLimit = 1000


wallLines = [
    [(leftLimit, bottomLimit), (leftLimit, topLimit)],
    [(rightLimit, bottomLimit), (rightLimit, topLimit)],
    [(leftLimit, bottomLimit), (rightLimit, bottomLimit)],
    [(leftLimit, topLimit), (rightLimit, topLimit)]
]

lines = createLines(nVisionLines, visionLength, fieldOfVision, wolfRotation, wolfPosition)
wallVisionLines = createLines(nVisionLines, visionLength, fieldOfVision, wolfRotation, wolfPosition)


def getWolfVision(wolfIndexPassed):

    wolf = animals[wolfIndexPassed - 1];
    currentPosition = (wolf.posX, wolf.posY)

    lines = createLines(nVisionLines, visionLength, fieldOfVision, wolf.rot, currentPosition)

    wallVisionLines = createLines(nVisionLines, visionLength, fieldOfVision, wolf.rot, currentPosition)

    wolfVision = []

    #TODO multiple sheep?   - like wolves but animals[i].animal == 2

    #lines towards sheep
    for lineIndex in range(nVisionLines):
        dist = distanceToObject(radius, sheepPosition, lines[lineIndex][0], lines[lineIndex][1], visionLength);

        partOFMaxDistance = dist/float(visionLength + radius);
        partOFMaxDistance = max(0, partOFMaxDistance)
        partOFMaxDistance = min(1, partOFMaxDistance)

        if(partOFMaxDistance > 0):
            partOFMaxDistance = 1 - partOFMaxDistance;

        wolfVision.append(partOFMaxDistance)

    #lines towards wolf
    for lineIndex in range(nVisionLines):

        maxPartOfMaxDistance = 0;

        for wolfIndex in range(len(animals)):

            #not self and is wolf
            if( wolfIndex != wolfIndexPassed and animals[wolfIndex].animal == 1):

                wolfPosition = (animals[wolfIndex].posX, animals[wolfIndex].posY)

                xDelta = currentPosition[0]-wolfPosition[0]
                yDelta = currentPosition[1]-wolfPosition[1]

                distance = math.sqrt(xDelta**2 + yDelta**2)
                if (distance < visionLength + radius):
                    dist = distanceToObject(radius, wolfPosition, lines[lineIndex][0], lines[lineIndex][1], visionLength);

                    partOFMaxDistance = dist/float(visionLength + radius);
                    partOFMaxDistance = max(0, partOFMaxDistance)
                    partOFMaxDistance = min(1, partOFMaxDistance)

                    if(partOFMaxDistance > 0):
                        partOFMaxDistance = 1 - partOFMaxDistance;

                    maxPartOfMaxDistance = max(maxPartOfMaxDistance, partOFMaxDistance)

        wolfVision.append(maxPartOfMaxDistance)

    #lines towards walls
    for lineIndex in range(nVisionLines):
        dist = distanceFromWall(wallVisionLines[lineIndex]);

        if(dist < 0 or dist > visionLength):
            wolfVision.append(0)
        else:
            #hit and is < visionlength from orgin

            partOFMaxDistance = dist/float(visionLength);
            partOFMaxDistance = min(1, partOFMaxDistance)
            partOFMaxDistance = max(0, partOFMaxDistance)

            if(partOFMaxDistance > 0):
                partOFMaxDistance = 1 - partOFMaxDistance;

            wolfVision.append(partOFMaxDistance)

    return wolfVision;


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
        self.reward = 0
        self.done = False
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
            self.animal = 2
            self.rotationSpeed = 1.0 / 6


    def move(self):
        rotTemp = self.rot%360
        speedX = math.cos(rotTemp * math.pi/180)*self.speed
        speedY = math.sin(rotTemp * math.pi/180)*self.speed

        self.posX += speedX
        self.posY += speedY

        self.posX = max(self.posX, leftLimit)
        self.posX = min(self.posX, rightLimit)

        self.posY = max(self.posY, bottomLimit)
        self.posY = min(self.posY, topLimit)

    def inputChange(self):
        #a = NeuralNetwork()
        #a.funca()
        if(self.animal == 1):
            self.speedChange((random.randrange(-10, 10)/10.0))
            self.rotationChange(random.randrange(-3600, 3600)/10.0)

            vision = getWolfVision(self.index)

            if(debug):
                print("x: " + str(self.posX))
                print("y: " + str(self.posY))
                print("r: " + str(self.rot))


            if(debug):
                for i in range(81):
                    value = vision[i]
                    if (value != 0):
                        input = "unknown"
                        if(i < 27):
                            input = "sheep: "
                        elif(i < 54):
                            input = "wolf: "
                        else:
                            input = "Wall: "
                        print(input + str(vision[i]))

                print("_______________________________________________")

        elif(self.animal == 2):
            self.speedChange((random.randrange(-10, 10)/10.0))
            self.rotationChange(random.randrange(-3600, 3600)/10.0)

            sheepPosition[0] = self.posX
            sheepPosition[1] = self.posY

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

    def getCurrentState(self):
        cState = getWolfVision(self.index)
        cState.append(self.speed/self.maxspeed)
        cState.append((self.rot%360)/360)

    def getNextState(self):
        if(nextReady == 1):
            return self.getCurrentState()

    def getReward(self):
        return self.getReward()
    def done(self):
        return self.getDone()

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

    def gameReset(self):
        self.gen = self.gen + 1
        self.animals = []

        self.animals.append(Entity("Sheep", 50, 50, 0, 270, 15, 10, 1))
        self.animals.append(Entity("Wolf", 100, 0, 0, 270, 20, 10, 2))
        self.animals.append(Entity("Wolf", 0, 100, 0, 270, 20, 10, 3))
        self.animals.append(Entity("Wolf", 350, 350, 0, 270, 20, 10, 4))
        self.saveFile += "\n"

    def nextState(self):
        self.index = self.index + 1
        self.saveFile += "T"
        for i in range (len(self.animals)):
            self.animals[i].inputChange();
            self.animals[i].move();
            self.saveFile += self.animals[i].asString()

    def done(self):
        endFile = ""

        for i in range(len(self.animals)):
            self.animals[i].inputChange();
            self.animals[i].move();
            self.saveFile += self.animals[i].asString()
            endFile += self.animals[i].objectInf()

        endFile += "B"
        endFile += self.saveFile

        with open("Output.txt", "w") as text_file:
            text_file.write(endFile)




