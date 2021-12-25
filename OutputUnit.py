import math
import jittor as jt
from jittor import Module


class OutputUnit(Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        k = math.sqrt(1 / output_size)
        self.W = jt.init.uniform((self.input_size, self.output_size), 'float32', -k, k)
        self.b = jt.init.uniform((self.output_size), 'float32', -k, k)

    def execute(self, x, finished=0):
        out = jt.nn.matmul(x, self.W) + self.b
        out[jt.where(finished)[0]] = 0
        return out
