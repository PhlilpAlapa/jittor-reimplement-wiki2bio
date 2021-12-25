import math
import jittor as jt
from jittor import Module


class fgateLstmUnit(Module):
    def __init__(self, hidden_size, input_size, field_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.field_size = field_size
        
        k = math.sqrt(1 / hidden_size)
        self.Wi = jt.init.uniform((self.input_size, 4 * self.hidden_size),'float32', -k, k)
        self.bi = jt.init.uniform((4 * self.hidden_size), 'float32', -k, k)
        self.Wh = jt.init.uniform((self.hidden_size, 4 * self.hidden_size), 'float32', -k, k)
        self.bh = jt.init.uniform((4 * self.hidden_size), 'float32', -k, k)
        self.Wf = jt.init.uniform((self.field_size, 2 * self.hidden_size), 'float32', -k, k)
        self.bf = jt.init.uniform((2 * hidden_size), 'float32', -k, k)

    def execute(self, x, fd, s, finished=0):
        h_prev, c_prev = s  # batch * hidden_size

        x = jt.nn.matmul(x, self.Wi) + jt.nn.matmul(h_prev, self.Wh) + self.bi + self.bh
        i, f, g, o = jt.split(x, x.shape[1]//4, 1)

        fd = jt.nn.matmul(fd, self.Wf) + self.bf
        r, d = jt.split(fd, fd.shape[1]//2, 1)
        
        c = jt.sigmoid(f) * c_prev + jt.sigmoid(i) * jt.tanh(g) + jt.sigmoid(r) * jt.tanh(d)
        h = jt.sigmoid(o) * jt.tanh(c)

        out, state = h, (h, c)
        cond = jt.where(finished)[0]
        out[cond] = 0
        state[0][cond] = h_prev[cond]
        state[1][cond] = c_prev[cond]

        return out, state
