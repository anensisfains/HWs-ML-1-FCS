from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

from typing import Optional


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
        self,
        early_stopping_rounds: int | None = 10,
        base_model_class = DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        subsample: float | int = 0.6,
        bagging_temperature: float | int = 1.0,
        bootstrap_type: str | None = 'Bernoulli',
        goss: bool | None = True,
        goss_k: float | int = 0.2,
        subsample_goss: float | int = 0.3, 
        rsm: float | int | None = None,
        quantization_type: str | None = None,
        nbins: int = 255
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params
        self.early_stopping_rounds = early_stopping_rounds

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate

        self.history = defaultdict(list) # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z) # подсмотрел в прошлогодней домашке хи-хи :)
        
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type
        
        self.goss = goss
        self.goss_k = goss_k
        self.subsample_goss = subsample_goss
        self.rsm = rsm
        self.quantization_type = quantization_type
        self.nbins = nbins
        self.rsm_history = []

    def partial_fit(self, X, y, preds):
        if self.bootstrap_type == 'Bernoulli':
            idx = np.arange(X.shape[0])
            batch_size = round(self.subsample * X.shape[0])
            idx_batch = np.random.choice(idx, size = batch_size, replace = False)
            X_batch = X[idx_batch]
            y_batch = y[idx_batch]
        elif self.bootstrap_type == 'Bayesian':
            x_random = np.random.uniform(0, 1, size = X.shape[0])
            weights = (-np.log2(x_random)) ** (self.bagging_temperature) 
            weights_prob = weights / np.sum(weights)
            idx = np.arange(X.shape[0])
            batch_size = round(self.subsample * X.shape[0])
            idx_batch = np.random.choice(idx, size = batch_size, replace = True, p = weights_prob)
            X_batch = X[idx_batch]
            y_batch = y[idx_batch]
            
        else:
            X_batch = X
            y_batch = y
            idx_batch = np.arange(X.shape[0])         
            
        # считаем сдвиги
        s_i = - self.loss_derivative(y_batch, preds[idx_batch])
        if self.goss:
            # берем индексы goss_k обьектов с максимальным градиентом
            sorted_s = np.argsort(np.abs(s_i))[::-1][:round(self.goss_k * len(s_i))]
            
            # берем индексы оставшихся элементов
            s_left = np.argsort(np.abs(s_i))[::-1][round(self.goss_k * len(s_i)):]
            # рандомно семплируем какую-то часть из оставшихся объектов
            idx = np.arange(len(s_left))
            batch_size_s = round(self.subsample_goss * X.shape[0])
            idx_batch_s = np.random.choice(idx, size = batch_size_s, replace = False)
            # сохранили индексы оставшихся элементов
            s_left = s_left[idx_batch_s]
            s_total = np.append(sorted_s, s_left)
            # оставшиеся элементы умножаем на вес
            s_i[s_left] = s_i[s_left] * (1 - self.goss_k) / self.subsample_goss
            # наконец сохраняем сдвиги и подвыборку
            s_i = s_i[s_total]
            X_batch = X[s_total]
            
        if self.rsm != None:
            idx_variables = np.arange(X_batch.shape[1])
            size_variables = round(self.rsm * X_batch.shape[1])
            idx_sample = np.random.choice(idx_variables, size_variables, replace = False)
            X_batch = X_batch[:, idx_sample]
        else:
            idx_sample = np.arange(X_batch.shape[1])
            
        
        # обучаем на них модель
        model = self.base_model_class(**self.base_model_params)
        model.fit(X_batch, s_i)
        new_preds = model.predict(X[:, idx_sample])
        optimal_gamma = self.find_optimal_gamma(y, preds, new_preds)
        
        #сохраняем модель (b_n) и гамму
        self.models.append(model)
        self.gammas.append(optimal_gamma)
        self.rsm_history.append(idx_sample)
        
        # добавляем новую модель в ансамбль
        self.train_predictions += optimal_gamma * self.learning_rate * new_preds

    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool 
        """
        X_train_fit = X_train.copy()
        X_val_fit = X_val.copy() if X_val != None else None
        if self.quantization_type != None:
            for i in range(X_train_fit.shape[1]):
                X_train_fit[:, i] = self.quantize(self.quantization_type, X_train_fit[:, i], self.nbins)
            if X_val != None:
                for i in range(X_val.shape[1]):
                    X_val_fit[:, i] = self.quantize(self.quantization_type, X_val_fit[:, i], self.nbins)
            
        self.train_predictions = np.zeros(y_train.shape[0])
        if X_val != None:
            self.val_predictions = np.zeros(y_val.shape[0])
        # теперь строим n базовых моделей, сохраняем нужные метрики
        if self.early_stopping_rounds != None:
            num_rounds = self.early_stopping_rounds 
            c = 0
        for i in range(self.n_estimators):
            self.partial_fit(X_train_fit, y_train, self.train_predictions)
            
            self.history['train_roc_auc'].append(roc_auc_score(y_train, self.train_predictions))
            self.history['train_loss'].append(self.loss_fn(y_train, self.train_predictions))
            
            if X_val != None:
                new_preds_val = self.models[i].predict(X_val_fit[:, self.rsm_history[i]])
                self.val_predictions += self.gammas[i] * self.learning_rate * new_preds_val
                self.history['val_roc_auc'].append(roc_auc_score(y_val, self.val_predictions))
                self.history['val_loss'].append(self.loss_fn(y_val, self.val_predictions))
                
                                                        # если i = 0 мы не сможем найти предыдущий элемент
                if (self.early_stopping_rounds != None) and (i != 0): 
                    if self.history['val_loss'][i] > self.history['val_loss'][i-1]:
                        c += 1
                        if c >= num_rounds:
                            break
                    else: # иначе обнуляем счетчик
                        c = 0
                        
        if plot:
            self.plot_history()

    def predict_proba(self, X):
        ensemble = np.zeros(X.shape[0])
        for i in range(len(self.models)):
            gamma = self.gammas[i]
            model = self.models[i]
            pred = model.predict(X)
            ensemble += gamma * self.learning_rate * pred
        prob_pos = self.sigmoid(ensemble)
        prob_neg = 1 - prob_pos
        return np.array(list(zip(prob_neg, prob_pos)))

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)
        
    def plot_history(self):
        """
        :param X: features array (any set)
        :param y: targets array (any set)
        """
        if 'val_roc_auc' in self.history:
            plt.figure(figsize = (16, 6))
            plt.suptitle('Графики изменения разных метрик на трейне и валидации', fontsize = 14)
        
            plt.subplot(1, 2, 1)
            plt.title('Трейн')
            train_roc_auc = self.history['train_roc_auc']
            train_loss = self.history['train_loss']
            plt.plot(train_roc_auc, label = 'ROC-AUC')
            plt.plot(train_loss, label = 'Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.title('Валидация')
            val_roc_auc = self.history['val_roc_auc']
            val_loss = self.history['val_loss']
            plt.plot(val_roc_auc, label = 'ROC-AUC')
            plt.plot(val_loss, label = 'Loss')
            plt.legend()
            
        else:   
            plt.figure(figsize = (12, 8))
            plt.title('График изменения разных метрик на трейне', fontsize = 14)
        
            train_roc_auc = self.history['train_roc_auc']
            train_loss = self.history['train_loss']
            plt.plot(train_roc_auc, label = 'ROC-AUC')
            plt.plot(train_loss, label = 'Loss')
            plt.legend()
    def quantize(self, quantization_type, x_array, nbins):
        x_array = x_array.toarray().reshape(-1)
        if len(np.unique(x_array)) == 1:
            return x_array
        
        if quantization_type == 'Uniform':
            thresholds = np.arange(x_array.min(), x_array.max(), (x_array.max() - x_array.min()) / nbins)
            x_quantized = np.digitize(x_array, thresholds)
        elif quantization_type == 'Quantile':
            quantiles = np.percentile(x_array, np.linspace(0, 100, nbins + 1))
            x_quantized = np.digitize(x_array, quantiles)
        return x_quantized
    def feature_importances(self):
        importances = []
        for model in self.models:
            importances.append(model.feature_importances_)
        importances = np.array(importances)
        importances_mean = importances.mean(axis = 0)
        return importances_mean / importances_mean.sum()