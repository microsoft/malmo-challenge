# Copyright (c) 2017 Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================================================================

from __future__ import absolute_import

from tkinter import Tk

from . import BaseAgent
from ..environment import VideoCapableEnvironment

FPS_KEYS_MAPPING = {'w': 'move 1', 'a': 'strafe -1', 's': 'move -1', 'd': 'strafe 1', ' ': 'jump 1',
                    'q': 'strafe -1', 'z': 'move 1'}

ARROW_KEYS_MAPPING = {'Left': 'turn -1', 'Right': 'turn 1', 'Up': 'move 1', 'Down': 'move -1'}

CONTINUOUS_KEYS_MAPPING = {'Shift_L': 'crouch 1', 'Shift_R': 'crouch 1',
                           '1': 'hotbar.1 1', '2': 'hotbar.2 1', '3': 'hotbar.3 1', '4': 'hotbar.4 1',
                           '5': 'hotbar.5 1',
                           '6': 'hotbar.6 1', '7': 'hotbar.7 1', '8': 'hotbar.8 1', '9': 'hotbar.9 1'} \
    .update(ARROW_KEYS_MAPPING)

DISCRETE_KEYS_MAPPING = {'Left': 'turn -1', 'Right': 'turn 1', 'Up': 'move 1', 'Down': 'move -1',
                         '1': 'hotbar.1 1', '2': 'hotbar.2 1', '3': 'hotbar.3 1', '4': 'hotbar.4 1', '5': 'hotbar.5 1',
                         '6': 'hotbar.6 1', '7': 'hotbar.7 1', '8': 'hotbar.8 1', '9': 'hotbar.9 1'}


class GuiAgent(BaseAgent):
    def __init__(self, name, environment, keymap, win_name="Gui Agent", size=(640, 480), visualizer=None):
        assert isinstance(keymap, list), 'keymap should be a list[character]'
        assert isinstance(environment, VideoCapableEnvironment), 'environment should inherit from BaseEnvironment'

        super(GuiAgent, self).__init__(name, environment.available_actions, visualizer)

        if not environment.recording:
            environment.recording = True

        self._env = environment
        self._keymap = keymap
        self._tick = 20

        self._root = Tk()
        self._root.wm_title = win_name
        self._root.resizable(width=False, height=False)
        self._root.geometry = "%dx%d" % size

        self._build_layout(self._root)

    def act(self, new_state, reward, done, is_training=False):
        pass

    def show(self):
        self._root.mainloop()

    def _build_layout(self, root):
        """
        Build the window layout
        :param root:
        :return:
        """
        raise NotImplementedError()

    def _get_keymapping_help(self):
        return self._keymap
