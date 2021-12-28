from pynput import mouse
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController

import numpy as np
from classifier.GestureLabels import GestureLabels

import platform

MVMT_RATIO = 0.5
MVMT_SCALE = 2
PIXELS_PER_SCROLL = 10
PIXELS_PER_THREE_HORIZONTAL = 200
PIXELS_PER_THREE_OR_FOUR_VERTICAL = 75
PIXELS_PER_FOUR_HORIZONTAL = 200

class GestureHandler():

    def __init__(self, mouse, keyboard) -> None:
        self.mouse = mouse
        self.keyboard = keyboard

class ScrollGestureHandler(GestureHandler):
    
    scrollDist = np.array([0, 0], dtype=np.float64)
    scrolled = False

    def start(self):
        self.scrollDist *= 0 # Set np array to 0. 
        self.scrolled = False
    
    def end(self):
        self.start()

    def changed(self, dDist):
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

    def did_activate(self):
        return self.scrolled

class ThreeGestureHandler(GestureHandler):
     
    dist = np.array([0, 0], dtype=np.float64)
    horizontaled = False
    verticaled = False

    def __down__(self):
        # Show desktop. 
        if platform.system() == 'Windows':
            self.keyboard.press(Key.cmd)
            self.keyboard.tap('d')
            self.keyboard.release(Key.cmd)
        elif platform.system() == 'Darwin':
            self.keyboard.tap(Key.f11)
    
    def __up__(self):
        # Show desktop switcher or mission control. 
        if platform.system() == 'Windows':
            self.keyboard.press(Key.cmd)
            self.keyboard.tap(Key.tab)
            self.keyboard.release(Key.cmd)
        elif platform.system() == 'Darwin':
            self.keyboard.press(Key.ctrl)
            self.keyboard.tap(Key.up)
            self.keyboard.release(Key.ctrl)

    def start(self):
        self.dist *= 0 # Set np array to 0. 
        self.horizontaled = False
        self.verticaled = False
    
    def end(self):
        if self.horizontaled:
            if platform.system() == 'Windows':
                self.keyboard.release(Key.alt)
            elif platform.system() == 'Darwin':
                self.keyboard.release(Key.cmd)
            
        self.start()

    def changed(self, dDist):
        self.dist += dDist

        # Scroll only when scrolled at least one scrolling step. 
        if (not self.horizontaled) and (not self.verticaled):
            if np.abs(self.dist[0]) >= PIXELS_PER_THREE_HORIZONTAL:
                self.horizontaled = True
                if platform.system() == 'Windows':
                    self.keyboard.press(Key.alt)
                elif platform.system() == 'Darwin':
                    self.keyboard.press(Key.cmd)
            elif np.abs(self.dist[1]) >= PIXELS_PER_THREE_OR_FOUR_VERTICAL:
                self.verticaled = True
                if self.dist[1] > 0:
                    self.__down__()
                else:
                    self.__up__()
        
        # Scroll in steps (similar to how scrolling wheel on the mouse has steps). 
        # Note: The unit of scrolling steps is unknown and depended on the operating system. 
        if self.horizontaled:
            steps = self.dist // PIXELS_PER_THREE_HORIZONTAL
            self.dist %= PIXELS_PER_THREE_HORIZONTAL
            if steps[0] > 0:
                self.keyboard.tap(Key.tab)
            elif steps[0] < 0:
                self.keyboard.press(Key.shift)
                self.keyboard.tap(Key.tab)
                self.keyboard.release(Key.shift)
            
    def did_activate(self):
        return self.horizontaled or self.verticaled


class FourGestureHandler(GestureHandler):
     
    dist = np.array([0, 0], dtype=np.float64)
    horizontaled = False
    verticaled = False

    def __down__(self):
        # Show desktop. 
        if platform.system() == 'Windows':
            self.keyboard.press(Key.cmd)
            self.keyboard.tap('d')
            self.keyboard.release(Key.cmd)
        elif platform.system() == 'Darwin':
            self.keyboard.tap(Key.f11)
    
    def __up__(self):
        # Show desktop switcher or mission control. 
        if platform.system() == 'Windows':
            self.keyboard.press(Key.cmd)
            self.keyboard.tap(Key.tab)
            self.keyboard.release(Key.cmd)
        elif platform.system() == 'Darwin':
            self.keyboard.press(Key.ctrl)
            self.keyboard.tap(Key.up)
            self.keyboard.release(Key.ctrl)

    def start(self):
        self.dist *= 0 # Set np array to 0. 
        self.horizontaled = False
        self.verticaled = False
    
    def end(self):
        if self.horizontaled:
            if platform.system() == 'Windows':
                self.keyboard.release(Key.cmd)
                self.keyboard.release(Key.ctrl)
            elif platform.system() == 'Darwin':
                self.keyboard.release(Key.ctrl)
            
        self.start()

    def changed(self, dDist):
        self.dist += dDist

        # Scroll only when scrolled at least one scrolling step. 
        if (not self.horizontaled) and (not self.verticaled):
            if np.abs(self.dist[0]) >= PIXELS_PER_FOUR_HORIZONTAL:
                self.horizontaled = True
                if platform.system() == 'Windows':
                    self.keyboard.press(Key.cmd)
                    self.keyboard.press(Key.ctrl)
                elif platform.system() == 'Darwin':
                    self.keyboard.press(Key.ctrl)
            elif np.abs(self.dist[1]) >= PIXELS_PER_THREE_OR_FOUR_VERTICAL:
                self.verticaled = True
                if self.dist[1] > 0:
                    self.__down__()
                else:
                    self.__up__()
        
        # Scroll in steps (similar to how scrolling wheel on the mouse has steps). 
        # Note: The unit of scrolling steps is unknown and depended on the operating system. 
        if self.horizontaled:
            steps = self.dist // PIXELS_PER_FOUR_HORIZONTAL
            self.dist %= PIXELS_PER_FOUR_HORIZONTAL
            if steps[0] > 0:
                self.keyboard.tap(Key.right)
            elif steps[0] < 0:
                self.keyboard.tap(Key.left)
            
    def did_activate(self):
        return self.horizontaled or self.verticaled


class Touchpad():

    mouse = MouseController()
    keyboard = KeyboardController()
    scrollHandler = ScrollGestureHandler(mouse, keyboard)
    threeHandler = ThreeGestureHandler(mouse, keyboard)
    fourHandler = FourGestureHandler(mouse, keyboard)

    isLeftPressed = False
    isRightPressed = False
    startScroll = False
    isThreePressed = False
    isFourPressed = False

    def __call__(self, gesture, dDist):
        noMvmt = gesture == GestureLabels.FIST
        leftDown = gesture == GestureLabels.LEFT_CLICK
        rightDown = gesture == GestureLabels.RIGHT_CLICK
        threeDown = gesture == GestureLabels.THREE
        fourDown = gesture == GestureLabels.FOUR

        '''
        leftDown is for left click or drag, right Down is for right click or scroll. 
        '''
        dDist *= MVMT_RATIO
        dDist = (np.abs(dDist) ** MVMT_SCALE) * np.sign(dDist)
        dDist = np.round(dDist) # Important: Round to int to avoid drifting. 

        # Move the mouse when not right clicked or scrolling. 
        if not (noMvmt or self.isRightPressed or self.isThreePressed or self.isFourPressed):
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
            self.scrollHandler.start()
        elif self.isRightPressed and rightDown:
            self.scrollHandler.changed(dDist)
        elif self.isRightPressed and (not rightDown):
            if not self.scrollHandler.did_activate(): # Right click if not scrolled. 
                self.mouse.click(Button.right)

            self.isRightPressed = False
            self.scrollHandler.end()

        # Three finger gesture. 
        if (not self.isThreePressed) and threeDown:
            self.isThreePressed = True
            self.threeHandler.start()
        elif self.isThreePressed and threeDown:
            self.threeHandler.changed(dDist)
        elif self.isThreePressed and (not threeDown):
            self.isThreePressed = False
            self.threeHandler.end()

        # Four finger gesture. 
        if (not self.isFourPressed) and fourDown:
            self.isFourPressed = True
            self.fourHandler.start()
        elif self.isFourPressed and fourDown:
            self.fourHandler.changed(dDist)
        elif self.isFourPressed and (not fourDown):
            self.isFourPressed = False
            self.fourHandler.end()



       