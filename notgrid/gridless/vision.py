import numpy as np
import math
import cocos

import cocos
from cocos.director import director
from cocos import draw




def distanceToObject(radius, objPos, line1, line2):

    x = np.array(objPos)

    u = np.array(line1)
    v = np.array(line2)

    n = v - u
    n /= np.linalg.norm(n, 2)

    intersectPos = u + n*np.dot(x - u, n)

    distanceLineToObject = math.sqrt( (intersectPos[0] - objPos[0])**2 + (intersectPos[1] - objPos[1])**2 )


    lineLength = math.sqrt( (line1[0] - line2[0])**2 + (line1[1] - line2[1])**2 )
    distToline1 = math.sqrt( (line1[0] - objPos[0])**2 + (line1[1] - objPos[1])**2 )
    distToline2 = math.sqrt( (line2[0] - objPos[0])**2 + (line2[1] - objPos[1])**2 )

    if(distanceLineToObject <= radius):
        orginToIntersectDistance = math.sqrt( (line1[0] - intersectPos[0])**2 + (line1[1] - intersectPos[1])**2 )

        if(orginToIntersectDistance < lineLength and distToline1 <= lineLength + radius and distToline2 <= lineLength + radius):
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

    length = len(wallLines);

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

    return closestDistance;

radius = 30
objPos = (240.0, 175.0)
visionLength = 100
fieldOfVision = 270
nVisionLines = 27
wolfPosition = (350, 200)
wolfRotation = -30


wallLines = [
    [(10, 10), (10, 400)],
    [(400, 10), (400, 400)],
    [(10, 10), (400, 10)],
    [(10, 400), (400, 400)]
]

x = line_intersection( (wallLines[0][0], wallLines[0][1]), (wallLines[1][0], wallLines[1][1]))

lines = createLines(nVisionLines, visionLength, fieldOfVision, wolfRotation, wolfPosition)
wallVisionLines = createLines(nVisionLines, visionLength, fieldOfVision, wolfRotation, wolfPosition)

sheepLines = createLines(200, radius, 360, 0, objPos)

wolfVision = []

class TestLayer(cocos.layer.Layer):
    def __init__(self):
        super().__init__()

        #lines towards sheep
        for lineIndex in range(len(lines)):
            dist = distanceToObject(radius, objPos, lines[lineIndex][0], lines[lineIndex][1]);



            if(dist < 0):
                #no hit
                line = draw.Line((lines[lineIndex][0][0], lines[lineIndex][0][1]), (lines[lineIndex][1][0], lines[lineIndex][1][1]),
                (255, 255, 255, 64))
            else:
                #hit
                line = draw.Line((lines[lineIndex][0][0], lines[lineIndex][0][1]), (lines[lineIndex][1][0], lines[lineIndex][1][1]),
                (0, 255, 0, 255))

            self.add(line)


        #lines towards walls
        for lineIndex in range(len(wallVisionLines)):
            dist = distanceFromWall(wallVisionLines[lineIndex]);

            if(dist < 0 or dist > visionLength):
                #no hit
                line = draw.Line((wallVisionLines[lineIndex][0][0], wallVisionLines[lineIndex][0][1]), (wallVisionLines[lineIndex][1][0], wallVisionLines[lineIndex][1][1]),
                (255, 255, 255, 64))
            else:
                #hit and is < visionlength from orgin

                x = wallVisionLines[lineIndex][1][0];
                y = wallVisionLines[lineIndex][1][1];


                line = draw.Line((wallVisionLines[lineIndex][0][0], wallVisionLines[lineIndex][0][1]), (wallVisionLines[lineIndex][1][0], wallVisionLines[lineIndex][1][1]),
                ( 255 , 0, 0, 255))

            self.add(line)

        #draw sheep

        for lineIndex in range(len(sheepLines)):
            line = draw.Line((sheepLines[lineIndex][0][0], sheepLines[lineIndex][0][1]), (sheepLines[lineIndex][1][0], sheepLines[lineIndex][1][1]),
            (0, 255, 255, 255))

            self.add(line)

        line = draw.Line(wallLines[0][0], wallLines[0][1], (255, 255, 0, 255))
        self.add(line)
        line = draw.Line(wallLines[1][0], wallLines[1][1], (255, 255, 0, 255))
        self.add(line)
        line = draw.Line(wallLines[2][0], wallLines[2][1], (255, 255, 0, 255))
        self.add(line)
        line = draw.Line(wallLines[3][0], wallLines[3][1], (255, 255, 0, 255))
        self.add(line)

def main():
    director.init()
    test_layer = TestLayer()
    main_scene = cocos.scene.Scene(test_layer)
    director.run(main_scene)


if __name__ == '__main__':
    main()
