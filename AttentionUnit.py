import math
import jittor as jt
from jittor import Module


class AttentionWrapper(Module):
    def __init__(self, hidden_size, input_size):
        self.hidden_size = hidden_size
        self.input_size = input_size

        k = math.sqrt(1 / hidden_size)
        self.Wh = jt.init.uniform((input_size, hidden_size),'float32', -k, k)
        self.bh = jt.init.uniform((hidden_size), 'float32', -k, k)
        self.Ws = jt.init.uniform((input_size, hidden_size), 'float32', -k, k)
        self.bs = jt.init.uniform((hidden_size), 'float32', -k, k)
        self.Wo = jt.init.uniform((2 * input_size, hidden_size), 'float32', -k, k)
        self.bo = jt.init.uniform((hidden_size), 'float32', -k, k)

    def execute(self, x, hs, finished=0):
        hs = jt.transpose(hs, [1, 0, 2])
        hs2d = jt.reshape(hs, [-1, self.input_size])
        phi_hs2d = jt.tanh(jt.nn.matmul(hs2d, self.Wh) + self.bh)
        phi_hs = jt.reshape(phi_hs2d, hs.shape)

        gamma_h = jt.tanh(jt.nn.matmul(x, self.Ws) + self.bs)
        weights = jt.reduce_add(phi_hs * gamma_h, 2, True)
        weights = jt.exp(weights - jt.max(weights, 0, True))
        weights = jt.divide(weights, 1e-6 + jt.reduce_add(weights, 0, True))
        context = jt.reduce_add(hs * weights, 0)
        out = jt.tanh(jt.nn.matmul(jt.concat([context, x], -1), self.Wo) + self.bo)
        out[jt.where(finished)[0]] = 0

        return out, weights
