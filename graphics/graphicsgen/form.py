from pyforms.basewidget import BaseWidget
from pyforms.controls   import ControlFile
from pyforms.controls   import ControlText
from pyforms.controls   import ControlSlider
from pyforms.controls   import ControlPlayer
from pyforms.controls   import ControlButton

class ComputerVisionAlgorithm(BaseWidget):

    def __init__(self, *args, **kwargs):
        super().__init__('Computer vision algorithm example')

        self._gen = ControlText('Gen')
        self._tick = ControlText('Tick')
        self._ticks = ControlText('Ticks')
        self._button1 = ControlButton('Start')
        self._button2 = ControlButton('Stop')
        self._button3 = ControlButton('Pause')
        self._info = ControlText('Info')

        #Define the organization of the Form Controls
        self._formset = [
            ('_gen', '_tick','_ticks'),
            ('_button1', '_button2', '_button3'),
            '_info'

        ]

        #Define the button action
        self._button1.value = self.__buttonAction

    def __buttonAction(self):
        """Button action event"""
        self._info.value =   self._gen.value +" "+   self._tick.value + " " +   self._ticks.value + " "



if __name__ == '__main__':

    from pyforms import start_app
    start_app(ComputerVisionAlgorithm)