import numpy as np
import math
import time
#import urllib2
from multiprocessing.dummy import Pool as ThreadPool

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
sheepPosition = (240.0, 175.0)
visionLength = 100
fieldOfVision = 270
nVisionLines = 27
wolfPosition = (350, 200)
wolfRotation = -30

wolfPositions = []
wolfPositions.append((350, 275))
wolfPositions.append((350, 150))

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

def appendVisionForIndex(lineIndex):
    returnObj = [lineIndex, 0, 0, 0]

    dist = distanceToObject(radius, sheepPosition, lines[lineIndex][0], lines[lineIndex][1]);

    partOFMaxDistance = dist/float(visionLength + radius);
    partOFMaxDistance = max(0, partOFMaxDistance)
    partOFMaxDistance = min(1, partOFMaxDistance)

    if(partOFMaxDistance > 0):
        partOFMaxDistance = 1 - partOFMaxDistance;

    returnObj[1] = partOFMaxDistance
    #wolfVision.append(partOFMaxDistance)

#lines towards wolf

    maxPartOfMaxDistance = 0;

    for wolfIndex in range(len(wolfPositions)):

        dist = distanceToObject(radius, wolfPositions[wolfIndex], lines[lineIndex][0], lines[lineIndex][1]);

        partOFMaxDistance = dist/float(visionLength + radius);
        partOFMaxDistance = max(0, partOFMaxDistance)
        partOFMaxDistance = min(1, partOFMaxDistance)

        if(partOFMaxDistance > 0):
            partOFMaxDistance = 1 - partOFMaxDistance;

        maxPartOfMaxDistance = max(maxPartOfMaxDistance, partOFMaxDistance)

    returnObj[2] = maxPartOfMaxDistance
    #wolfVision.append(maxPartOfMaxDistance)

#lines towards walls
    dist = distanceFromWall(wallVisionLines[lineIndex]);

    if(dist < 0 or dist > visionLength):
        pass
        #wolfVision.append(0)
    else:
        #hit and is < visionlength from orgin

        partOFMaxDistance = dist/float(visionLength);
        partOFMaxDistance = min(1, partOFMaxDistance)
        partOFMaxDistance = max(0, partOFMaxDistance)

        if(partOFMaxDistance > 0):
            partOFMaxDistance = 1 - partOFMaxDistance;

        returnObj[3] = maxPartOfMaxDistance
        #wolfVision.append(partOFMaxDistance)

    return returnObj

def calculateParallel(numbers, threads):
    pool = ThreadPool(threads)
    results = pool.map(appendVisionForIndex, numbers)
    pool.close()
    pool.join()
    return results


def main():
    #lines towards sheep

    method = "no_threads"

    wolfVision = []

    #TODO make for n number of lines
    numbers = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]

    if(method == "threads"):
        results = calculateParallel(numbers, 4)

        for n in range(len(results)):
            wolfVision.append(results[n][1])
            wolfVision.append(results[n][2])
            wolfVision.append(results[n][3])

    else:
        for n in range(27):
            results = appendVisionForIndex(n)
            wolfVision.append(results[1])
            wolfVision.append(results[2])
            wolfVision.append(results[3])



    for n in range(len(results)):
        pass
        #print(results[n])
        #wolfVision.append(results[n][1])
        #wolfVision.append(results[n][2])
        #wolfVision.append(results[n][3])



    #for i in range(len(wolfVision)):
    #    print("wolfVision " + str(i) + ":" + str(wolfVision[i]))

    #for lineIndex in range(nVisionLines):
    #    appendVisionForIndex(lineIndex)


    #for i in range(len(wolfVision)):
    #    print("wolfVision " + str(i) + ":" + str(wolfVision[i]))

if __name__ == '__main__':
    start_time = time.time()
    ticks = 100;

    print("testing iterations: " + str(ticks))
    for value in range(ticks):
        main()

    print("--- %s seconds ---" % (time.time() - start_time))
