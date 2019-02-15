import cocos
class tick:
    def __init__(self, string):

        tempStrings = str.split(string, "L")
        temp2String = str.split(str.replace(tempStrings[0], "O", ""), "[[")
        self.Index = temp2String[0]
        self.O = temp2String[1]
        self.L = tempStrings[1]


f = open("Output.txt", "r")
G = f.readlines()
print(str.split(G[0], "T")[1])
p1 = tick(str.split(G[0], "T")[1])
print(p1.O)
print(p1.L)


class HelloWorld(cocos.layer.Layer):
    def __init__(self):
        super(HelloWorld, self).__init__()
        label = cocos.text.Label(
            'Hello, world',
            font_name='Times New Roman',
            font_size=32,
            anchor_x='center', anchor_y='center'
        )
        self.add(label)
        cocos.director.director.init()

hello_layer = HelloWorld()
main_scene = cocos.scene.Scene(hello_layer)
cocos.director.director.run(main_scene)