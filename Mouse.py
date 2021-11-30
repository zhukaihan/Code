from pynput.mouse import Button, Controller
import numpy as np

MVMT_RATIO = 0.5
MVMT_SCALE = 2

class Mouse():

    mouse = Controller()
    isLeftPressed = False
    isRightPressed = False

    def __call__(self, leftDown, rightDown, dDist):
        dDist *= MVMT_RATIO
        dDist = (np.abs(dDist) ** MVMT_SCALE) * np.sign(dDist)

        self.mouse.move(dDist[0], dDist[1])

        if (not self.isLeftPressed) and leftDown and (not rightDown): # Prioritize right click, due to model landmark localization issue. 
            self.mouse.press(Button.left)
            self.isLeftPressed = True
        if (self.isLeftPressed) and (not leftDown):
            self.mouse.release(Button.left)
            self.isLeftPressed = False

        if (not self.isRightPressed) and rightDown:
            self.mouse.press(Button.right)
            self.isRightPressed = True
        if (self.isRightPressed) and (not rightDown):
            self.mouse.release(Button.right)
            self.isRightPressed = False

        # Double click; this is different from pressing and releasing
        # twice on macOS
        # mouse.click(Button.left, 2)
