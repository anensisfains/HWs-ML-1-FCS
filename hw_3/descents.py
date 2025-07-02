from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Template for calc_gradient function
        Calculate gradient of loss function with respect to weights
        :param x: features array
        :param y: targets array
        :return: gradient: np.ndarray
        """
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for x and y with our weights
        :param x: features array
        :param y: targets array
        :return: loss: float
        """
        if self.loss_function is LossFunction.MSE:
            return ((x @ self.w - y) ** 2).mean()
        
        elif self.loss_function is LossFunction.LogCosh:
            return np.log1p(np.cosh(x @ self.w - y) - 1).mean() # минус 1 т.к. в log1p добавляется 1
        
        elif self.loss_function is LossFunction.MAE:
            return np.abs(x @ self.w - y).mean()
        
        elif self.loss_function is LossFunction.Huber:
            loss_vec = np.zeros(x.shape[0])
            mask = np.abs(x @ self.w - y) < 2
            loss_vec[mask] = ((x @ self.w - y) ** 2)[mask]
            
            loss_vec[~mask] = np.abs(x @ self.w - y)[~mask]
            
            return loss_vec.mean()
        
        #raise NotImplementedError('BaseDescent calc_loss function not implemented')

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for x
        :param x: features array
        :return: prediction: np.ndarray
        """
        return x @ self.w
        #raise NotImplementedError('BaseDescent predict function not implemented')


class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights with respect to gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        w_new = self.w - (self.lr() * gradient) 
        diff = w_new - self.w
        self.w = w_new
        
        return diff
        
        #raise NotImplementedError('VanillaGradientDescent update_weights function not implemented')

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.loss_function is LossFunction.MSE:
            grad = (-2 / x.shape[0]) * (x.T @ (y - x @ self.w))
            return grad
        
        elif self.loss_function is LossFunction.LogCosh:
            grad = (1 / x.shape[0]) * (np.tanh(x @ self.w - y).T @ x)
            return grad
        
        elif self.loss_function is LossFunction.MAE:
            grad = (1 / x.shape[0]) * (x.T @ np.sign(x @ self.w - y))
            return grad
        
        elif self.loss_function is LossFunction.Huber:
            mask = np.abs(x @ self.w - y) < 2
            
            grad_mse = x[mask].T @ (x[mask] @ self.w - y[mask])
            grad_mae = 2 * x[~mask].T @ np.sign((x[~mask] @ self.w - y[~mask]))
            grad = (1 / x.shape[0]) * (grad_mse + grad_mae)
            return grad
        
        #raise NotImplementedError('VanillaGradientDescent calc_gradient function not implemented')


class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):
        """
        :param batch_size: batch size (int)
        """
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        batch_idx = np.random.randint(0, x.shape[0], self.batch_size)
        x_st = x[batch_idx, :]
        y_st = y[batch_idx]
        
        if self.loss_function is LossFunction.MSE:
            grad = (-2 / x_st.shape[0]) * (x_st.T @ (y_st - x_st @ self.w))
            return grad
        
        elif self.loss_function is LossFunction.LogCosh:
            grad = (1 / x_st.shape[0]) * (np.tanh(x_st @ self.w - y_st).T @ x_st)
            return grad
        
        elif self.loss_function is LossFunction.MAE:
            grad = (1 / x_st.shape[0]) * (x_st.T @ np.sign(x_st @ self.w - y_st))
            return grad
        
        elif self.loss_function is LossFunction.Huber:
            mask = np.abs(x_st @ self.w - y_st) < 2
            grad_mse = x_st[mask].T @ (x_st[mask] @ self.w - y_st[mask])
            grad_mae = 2 * x_st[~mask].T @ np.sign((x_st[~mask] @ self.w - y_st[~mask]))
            grad = (1 / x_st.shape[0]) * (grad_mse + grad_mae)
            return grad
        #raise NotImplementedError('StochasticDescent calc_gradient function not implemented')


class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights with respect to gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        h_new = (self.alpha * self.h) + (self.lr() * gradient)
        w_new = self.w - h_new
        diff = w_new - self.w
        
        self.h = h_new
        self.w = w_new
        
        return diff
        #raise NotImplementedError('MomentumDescent update_weights function not implemented')


class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights & params
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.iteration += 1 # сделал пояснение к этому моменту в тетрадке
          
        m_new = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        v_new = self.beta_2 * self.v + (1 - self.beta_2) * (gradient ** 2)
        m_hat = m_new / (1 - (self.beta_1 ** (self.iteration)))
        v_hat = v_new / (1 - (self.beta_2 ** (self.iteration)))
        w_new = self.w - ((self.lr() * m_hat) / (np.sqrt(v_hat) + self.eps))
        diff = w_new - self.w
        
        self.w = w_new
        self.m = m_new
        self.v = v_new
        
        return diff
        #raise NotImplementedError('Adagrad update_weights function not implemented')

        
class AMSGrad(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)
        self.v_hat_prev: np.ndarray = np.zeros(dimension) # новая переменная, вектор v_hat с прошлого шага
        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights & params
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.iteration += 1
          
        m_new = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        v_new = self.beta_2 * self.v + (1 - self.beta_2) * (gradient ** 2)
        m_hat = m_new / (1 - (self.beta_1 ** (self.iteration)))
        # считаем новое значение для v_hat
        v_hat = np.maximum(v_new, self.v_hat_prev)
        
        w_new = self.w - ((self.lr() * m_hat) / (np.sqrt(v_hat) + self.eps))
        diff = w_new - self.w
        
        self.w = w_new
        self.m = m_new
        self.v = v_new
        self.v_hat_prev = v_hat # дополнительная переменная, которую мы храним
        
        return diff
        #raise NotImplementedError('Adagrad update_weights function not implemented')

        
class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """
        l2_gradient: np.ndarray = np.zeros(x.shape[1])  
        grad = self.w                  # вообще здесь должно быть 2w, но так как мы умножаем просто на мю, то так
        l2_gradient[:-1] = grad[:-1]   # и конечно нам нельзя регуляризовать bias

        return super().calc_gradient(x, y) + l2_gradient * self.mu


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg,
        'AMSGrad': AMSGrad
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]
    return descent_class(**descent_config.get('kwargs', {}))
