import numpy as np
import math
import time

def distanceToObject(radius, objPos, line1, line2, visionLength):

    distToline1 = math.sqrt( (line1[0] - objPos[0])**2 + (line1[1] - objPos[1])**2 )
    distToline2 = math.sqrt( (line2[0] - objPos[0])**2 + (line2[1] - objPos[1])**2 )

    if(distToline1 > visionLength + radius or distToline2 > visionLength + radius):
        return -1;


    u = np.array(line1)

    n = np.array(line2) - u

    #n /= np.linalg.norm(n, 2)
    n /= visionLength

    intersectPos = u + n*np.dot(np.array(objPos) - u, n)

    if(math.sqrt( (intersectPos[0] - objPos[0])**2 + (intersectPos[1] - objPos[1])**2 ) <= radius):
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
            if( math.sqrt( (visionLine[1][0] - intersection[0])**2 + (visionLine[1][1] - intersection[1])**2 ) <= visionLength):
                distance = math.sqrt( (visionLine[0][0] - intersection[0])**2 + (visionLine[0][1] - intersection[1])**2 )

                closestDistance = min(closestDistance, distance)

    return closestDistance;

radius = 30
sheepPosition = (240.0, 175.0)
visionLength = 100
fieldOfVision = 270
nVisionLines = 27
wolfPosition = (350, 200)
wolfRotation = -30

wolfPositions = []
wolfPositions.append((350, 275))
wolfPositions.append((350, 150))
wolfPositions.append((125, 150))
wolfPositions.append((125, 382))
wolfPositions.append((113, 39))

wallLines = [
    [(10, 10), (10, 400)],
    [(400, 10), (400, 400)],
    [(10, 10), (400, 10)],
    [(10, 400), (400, 400)]
]

lines = createLines(nVisionLines, visionLength, fieldOfVision, wolfRotation, wolfPosition)
wallVisionLines = createLines(nVisionLines, visionLength, fieldOfVision, wolfRotation, wolfPosition)

sheepLines = createLines(150, radius, 360, 0, sheepPosition)

wolfVision = []

def getWolfVision(wolfIndexPassed):

    lines = createLines(nVisionLines, visionLength, fieldOfVision, wolfRotation, wolfPosition)
    wallVisionLines = createLines(nVisionLines, visionLength + 1, fieldOfVision + 1, wolfRotation, wolfPosition)

    wolfVision = []
    #lines towards sheep
    for lineIndex in range(nVisionLines):
        partOFMaxDistance = distanceToObject(radius, sheepPosition, lines[lineIndex][0], lines[lineIndex][1], visionLength)/float(visionLength + radius);
        partOFMaxDistance = max(0, partOFMaxDistance)
        partOFMaxDistance = min(1, partOFMaxDistance)

        if(partOFMaxDistance > 0):
            partOFMaxDistance = 1 - partOFMaxDistance;

        wolfVision.append(partOFMaxDistance)

    #lines towards wolf
    for lineIndex in range(nVisionLines):
        maxPartOfMaxDistance = 0;

        for wolfIndex in range(len(wolfPositions)):
            if(wolfIndex != wolfIndexPassed):

                distance = math.sqrt((wolfPositions[wolfIndexPassed][0]-wolfPositions[wolfIndex][0])**2 + (wolfPositions[wolfIndexPassed][1]-wolfPositions[wolfIndex][1])**2)
                if (distance < visionLength + radius):
                    dist = distanceToObject(radius, wolfPositions[wolfIndex], lines[lineIndex][0], lines[lineIndex][1], visionLength);

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
    #for i in range(len(wolfVision)):
    #    print("wolfVision " + str(i) + ":" + str(wolfVision[i]))

if __name__ == '__main__':
    start_time = time.time()
    ticks = 10000;

    print("testing iterations: " + str(ticks))
    for i in range(ticks):
        getWolfVision(0)

    print("--- %s seconds ---" % (time.time() - start_time))
