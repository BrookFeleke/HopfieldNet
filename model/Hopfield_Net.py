# models/hopfield_net.py

from typing import NamedTuple
import numpy as np
import numpy.typing as npt

class PredictResult(NamedTuple):
    states: npt.NDArray[np.int8]
    energies: npt.NDArray[np.float32]

class HopfieldNet:
    def __init__(self, inputs: npt.NDArray[np.int8]) -> None:
        """
        Initializes the Hopfield network with the given input patterns.

        The weight matrix is calculated as the outer product of the input patterns
        minus the mean of the patterns. The diagonal of the weight matrix is set to zero
        to prevent self-reinforcement.

        Args:
            inputs: A 2D numpy array of integers representing the input patterns.
        """
        self.dim = len(inputs[0])
        self.patterns = len(inputs)
        self.W = np.zeros((self.dim, self.dim))
        mean = np.sum([np.sum(i) for i in inputs]) / (self.patterns * self.dim)
        for i in range(self.patterns):
            t = inputs[i] - mean
            self.W += np.outer(t, t)
        np.fill_diagonal(self.W, 0)
        self.W /= self.patterns

    def energy(self, x: npt.NDArray[np.int8], bias: float) -> float:
        return -0.5 * np.dot(x.T, np.dot(self.W, x)) + np.sum(bias * x)

    def sync_predict(self, x: npt.NDArray[np.int8], bias: float) -> PredictResult:
        """
        Performs synchronous prediction on the input pattern using the Hopfield network.
    
        Args:
        x: A numpy array of integers representing the initial state of the pattern.
        bias: A float representing the bias term to be subtracted during state update.
        
        Returns:
        A PredictResult named tuple containing the list of states and corresponding energies
        at each iteration of the synchronous update process.
        """
        es = [self.energy(x, bias)]
        xs = [x]
        for _ in range(100):
            x_prev = xs[-1]
            x_new = np.sign(np.dot(self.W, x_prev) - bias)
            e_new = self.energy(x_new, bias)
            xs.append(x_new)
            es.append(e_new)
        return PredictResult(states=xs, energies=es)

    def async_predict(self, x: npt.NDArray[np.int8], bias: float) -> PredictResult:
        """
        Performs asynchronous prediction on the input pattern using the Hopfield network.

        Args:
        x: A numpy array of integers representing the initial state of the pattern.
        bias: A float representing the bias term to be subtracted during state update.
        
        Returns:
        A PredictResult named tuple containing the list of states and corresponding energies
        at each iteration of the asynchronous update process.
        """
        es = [self.energy(x, bias)]
        xs = [x]
        for i in range(len(x)):
            state = xs[-1].copy()
            state_i_new = np.sign(np.dot(self.W[i, :], state) - bias)
            state[i] = state_i_new
            xs.append(state)
            es.append(self.energy(state, bias))
        return PredictResult(states=xs, energies=es)

    def predict(self, x: npt.NDArray[np.int8], bias: float, sync: bool) -> PredictResult:
        return self.sync_predict(x, bias) if sync else self.async_predict(x, bias)
