import cocos
import time
from cocos      import euclid
from cocos    import draw
class Entities:
    def __init__(self, intText):
        self.entityList = []
        for infoText in str.split(intText, "O"):
            if (infoText != ""):
                self.entityList.append(Entity(infoText))


    def updateWithTick(self, string):
        var = 0
        for infoText in str.split(string, "O"):
            if (infoText != ""):
                self.entityList[var].tickChange(infoText)
            var += 1

    def getEntities(self):
        return self.entityList

class Entity:
    def __init__(self, intText):
        initValues = str.split(intText, "|")
        self.index = initValues[0]
        self.animal = initValues[1]
        sizeValues = str.split(initValues[2], ",")
        self.length = sizeValues[0]
        self.height = sizeValues[1]
        self.rot = 0
        self.posX = 0
        self.posY = 0

    def tickChange(self, string):
        changeValues = str.split(string, "|")
        posValues = str.split(changeValues[1], ",")
        if(changeValues[0] == self.index):
            self.posX = posValues[0]
            self.posY = posValues[1]
            self.rot = changeValues[2]


class ticks:
    def __init__(self, string):
        self.ticks = []
        for tickText in str.split(string, "T"):
            if (tickText != ""):
                self.ticks.append(tickText)

    def getTickInfo(self, tick):
        return self.ticks[tick]

class data:
    def __init__(self, string):
        self.generations = []
        for generationText in str.split(string, "\n"):
            if (generationText != ""):
                self.generations.append(ticks(generationText))

    def getTickInfo(self, gen, tick):
        return self.generations[gen].getTickInfo(tick)



f = open("Output.txt", "r")

text = str.split(f.read(), "B")

entities = Entities(text[0])

fullData = data(text[1])

entitiesState = entities.getEntities()

for x in range(0, 100):
    entities.updateWithTick(fullData.getTickInfo(0, x))
    entitiesState = entities.getEntities()
    print("PosX: " + str(entitiesState[0].posX) + " PosY: " + str(entitiesState[0].posY) + " Rot: " + str(entitiesState[0].rot))


print(fullData.getTickInfo(0, 1))
print(fullData.getTickInfo(0, 50))
print(fullData.getTickInfo(0, 100))



class HelloWorld(cocos.layer.Layer):
    def __init__(self, animals, pos):
        super(HelloWorld, self).__init__()
        self.label = cocos.text.Label(
            'Hello, world',
            font_name='Times New Roman',
            font_size=32,
            anchor_x='center', anchor_y='center'
        )
        self.label.position = pos, pos
        self.rect = cocos.sprite.Sprite("Wolf.png")
        self.rect.position = 320, 240
        self.rect.scale = 0.5

        self.add(self.rect)
        self.add(self.label)
    def draw(self):
        self.rect = 50, 50



cocos.director.director.init()

for x in range(0, 500):
    hello_layer = HelloWorld(entitiesState, x)
    main_scene = cocos.scene.Scene(hello_layer)
    cocos.director.director.run(main_scene)
    time.sleep(0.300)
    print("Update")

