from pynput.mouse import Button, Controller
import numpy as np

MVMT_RATIO = 0.5
MVMT_SCALE = 2
INIT_TIME_TO_SCROLL = 10
PIXELS_PER_SCROLL = 10

class Mouse():

    mouse = Controller()
    isLeftPressed = False
    isRightPressed = False
    startScroll = False

    def __call__(self, leftDown, rightDown, dDist):
        '''
        leftDown is for left click or drag, right Down is for right click or scroll. 
        '''
        dDist *= MVMT_RATIO
        dDist = (np.abs(dDist) ** MVMT_SCALE) * np.sign(dDist)
        dDist = np.round(dDist) # Important: Round to int to avoid drifting. 

        # Move the mouse when not right clicked or scrolling. 
        if not self.isRightPressed:
            self.mouse.move(dDist[0], dDist[1])

        # Left button. 
        if (not self.isLeftPressed) and leftDown and (not rightDown): # Prioritize right click, due to model landmark localization issue when index finger is occluded. 
            self.mouse.press(Button.left)
            self.isLeftPressed = True
        elif self.isLeftPressed and (not leftDown):
            self.mouse.release(Button.left)
            self.isLeftPressed = False
        # Future feature: double click. This is different from pressing and releasing twice on macOS
        # mouse.click(Button.left, 2)

        # Right button or scroll. 
        if (not self.isRightPressed) and rightDown:
            self.isRightPressed = True
            self.resetScroll()
        elif self.isRightPressed and rightDown:
            self.scroll(dDist)
        elif self.isRightPressed and (not rightDown):
            self.isRightPressed = False
            if not self.scrolled: # Right click if not scrolled. 
                self.mouse.click(Button.right)



    scrollDist = np.array([0, 0], dtype=np.float64)
    scrolled = False

    def resetScroll(self):
        self.scrollDist *= 0 # Set np array to 0. 
        self.scrolled = False

    def scroll(self, dDist):
        self.scrollDist += dDist

        # Scroll only when scrolled at least one scrolling step. 
        if (not self.scrolled) and np.any(np.abs(self.scrollDist) >= PIXELS_PER_SCROLL):
            self.scrolled = True
        
        # Scroll in steps (similar to how scrolling wheel on the mouse has steps). 
        # Note: The unit of scrolling steps is unknown and depended on the operating system. 
        if self.scrolled:
            steps = self.scrollDist // PIXELS_PER_SCROLL
            self.scrollDist %= PIXELS_PER_SCROLL
            self.mouse.scroll(steps[0], -steps[1]) # Vertical scroll's direction is the opposite. 
