"""
  Define some geometric objects:
  - ellipse
"""
import numpy as np


class Ellipse():
    def __init__(self):
        self.__x = None
        self.__y = None
        self.__height = None
        self.__width = None
        self.__angle = None

    def from_opencv_tuples(self, opencv):
        self.__x = opencv[0][0]
        self.__y = opencv[0][1]
        self.__height = opencv[1][0]
        self.__width = opencv[1][1]
        self.__angle = opencv[2]

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def height(self):
        return self.__height

    @property
    def width(self):
        return self.__width

    @property
    def angle(self):
        return self.__angle

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
