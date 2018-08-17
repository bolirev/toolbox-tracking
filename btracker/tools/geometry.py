"""
  Define some geometric objects:
  - ellipse
"""
import numpy as np


class Ellipse():
    def __init__(self, x=0, y=0,
                 height=1, width=1,
                 angle=0):
        self.__x = x
        self.__y = y
        self.__height = height
        self.__width = width
        self.__angle = angle

    def from_opencv_tuples(self, opencv):
        self.__x = opencv[0][0]
        self.__y = opencv[0][1]
        self.__height = opencv[1][0]
        self.__width = opencv[1][1]
        self.__angle = opencv[2]

    def from_ivtrace_series(self, pdseries):
        self.__x = pdseries.x
        self.__y = pdseries.y
        self.__angle = pdseries.orientation
        self.__width = np.sqrt((4*pdseries.size)/(np.pi*pdseries.roundness))
        self.__height = self.__width * pdseries.roundness

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, x):
        self.__x = x

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, y):
        self.__y = y

    @property
    def height(self):
        return self.__height

    @height.setter
    def height(self, h):
        self.__height = h

    @property
    def width(self):
        return self.__width

    @width.setter
    def width(self, w):
        self.__width = w

    @property
    def angle(self):
        return self.__angle

    @angle.setter
    def angle(self, a):
        self.__angle = a

    @property
    def roundness(self):
        return self.height / self.width

    @property
    def area(self):
        return np.pi * self.height * self.width / 4

    @property
    def spread(self):
        angle = np.deg2rad(self.angle)
        x_spread = 2 * np.sqrt(((self.height / 2) * np.cos(angle))**2 +
                               ((self.width / 2) * np.sin(angle))**2)
        y_spread = 2 * np.sqrt(((self.height / 2) * np.sin(angle))**2 +
                               ((self.width / 2) * np.cos(angle))**2)
        return x_spread, y_spread

    @property
    def center(self):
        return (self.x, self.y)

    @property
    def size(self):
        return (self.height, self.width)
