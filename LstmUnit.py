import jittor as jt
import math
from jittor import Module


class LstmUnit(Module):
    def __init__(self, hidden_size, input_size):
        self.hidden_size = hidden_size
        self.input_size = input_size

        k = math.sqrt(1 / hidden_size)
        self.Wi = jt.init.uniform((self.input_size, 4 * self.hidden_size), jt.float32, -k, k)
        self.bi = jt.init.uniform((4 * self.hidden_size), jt.float32, -k, k)
        self.Wh = jt.init.uniform((self.hidden_size, 4 * self.hidden_size), jt.float32, -k, k)
        self.bh = jt.init.uniform((4 * self.hidden_size), jt.float32, -k, k)

    def execute(self, x, s, finished=0):
        h_prev, c_prev = s
        
        x = jt.matmul(x, self.Wi) + jt.matmul(h_prev, self.Wh) + self.bi + self.bh
        
        i, f, g, o = jt.split(x, x.shape[1]//4, 1)

        # Final Memory cell
        c = jt.sigmoid(f) * c_prev + jt.sigmoid(i) * jt.tanh(g)
        h = jt.sigmoid(o) * jt.tanh(c)

        out, state = h, (h, c)
        cond = jt.where(finished)[0]
        out[cond] = 0
        state[0][cond] = h_prev[cond]
        state[1][cond] = c_prev[cond]

        return out, state
