from pyforms.basewidget import BaseWidget
from pyforms.controls   import ControlFile
from pyforms.controls   import ControlText
from pyforms.controls   import ControlSlider
from pyforms.controls   import ControlPlayer
from pyforms.controls   import ControlButton
from pyforms.controls   import ControlOpenGL
from pyforms.controls   import ControlImage
from OpenGL.GL import *
from OpenGL.GL.shaders import *
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc
import time


class ComputerVisionAlgorithm(BaseWidget):

    def __init__(self, *args, **kwargs):
        super().__init__('Computer vision algorithm example')

        self._gen = ControlText('Gen')
        self._tick = ControlText('Tick')
        self._ticks = ControlButton('Jump')
        self._button1 = ControlButton('Start')
        self._button2 = ControlButton('Stop')
        self._button3 = ControlButton('Pause')
        self._info = ControlText('Info')
        self._image = ControlImage('Image')
        self.width = 1000
        self.height = 1000
        self.FPS = 24
        self.seconds = 10

        #triangle = [-0.5, -0.5, 0,
        #            0.5, -0.5, 0,
        #                0, 0.5, 0]
#
        #vertex_shader ="""
        #in vec4 position;
        #void main()
        #{
        #    gl_Position = position;
        #}
        #"""
#
#
        #fragment_shader ="""
        #void main()
        #{
        #    gl_FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
        #}
        #"""
#
        #shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        #                                          OpenGL.GL.shader.compileShader(fragment_shader, GL_FRAGMENT_SHADER))
        #VBO = glGenBuffers(1)
        #glBindBuffer(GL_ARRAY_BUFFER, VBO)
        #glBufferData(GL_ARRAY_BUFFER, sizeof(triangle), triangle, GL_STATIC_DRAW)
#
        #position = glGetAttribLocation(shader, "position")
        #glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 0, None)
        #glEnableVertexAttribArray(position)
#
        #glUseProgram(shader)
#
#
#
        #self._opengl = ControlOpenGL()
#
#
        #glClearColor(0.,0.3,0.2,1.0)
        #Define the organization of the Form Controls
        self._formset = [
            ('_gen', '_tick','_ticks','_button1', '_button2', '_button3'),
            '_info',
            '_image'

        ]

        #Define the button action
        self._button1.value = self.__buttonAction

    def __buttonAction(self):
        """Button action event"""
        self._info.value =   self._gen.value +" "+   self._tick.value + " "
        for x in range(0,320) :
            print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))
            self.fourcc = VideoWriter_fourcc(*'MP42')
            self.frame = np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
            self._image.value = self.frame
            self._image.init_form()


        #glClear(GL_COLOR_BUFFER_BIT)
        #glDrawArrays(GL_TRIANGLES, 0,  3)




if __name__ == '__main__':

    from pyforms import start_app
    start_app(ComputerVisionAlgorithm)