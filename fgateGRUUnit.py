import math
import jittor as jt
from jittor import Module


class fgateGRUUnit(Module):
    def __init__(self, hidden_size, input_size, field_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.field_size = field_size

        k = math.sqrt(1 / hidden_size)
        self.Wi = jt.init.uniform(
            (self.input_size, 3 * self.hidden_size), 'float32', -k, k)
        self.bi = jt.init.uniform((3 * self.hidden_size), 'float32', -k, k)
        self.Wh = jt.init.uniform(
            (self.hidden_size, 3 * self.hidden_size), 'float32', -k, k)
        self.bh = jt.init.uniform((3 * self.hidden_size), 'float32', -k, k)
        self.Wf = jt.init.uniform(
            (self.field_size, 3 * self.hidden_size), 'float32', -k, k)
        self.bf = jt.init.uniform((3 * hidden_size), 'float32', -k, k)

    def execute(self, x, fd, h, finished=0):
        gi = jt.nn.matmul(x, self.Wi) + jt.nn.matmul(fd,
                                                     self.Wf) + self.bi + self.bf
        gh = jt.nn.matmul(h, self.Wh) + self.bh

        i_r, i_i, i_n = jt.split(gi, gi.shape[1]//3, 1)
        h_r, h_i, h_n = jt.split(gh, gh.shape[1]//3, 1)

        resetgate = jt.sigmoid(i_r + h_r)
        inputgate = jt.sigmoid(i_i + h_i)
        newgate = jt.tanh(i_n + resetgate * h_n)
        h_new = newgate + inputgate * (h - newgate)

        cond = jt.where(finished)[0]
        h_new[cond] = h[cond]

        return h_new
