import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector, min_samples_leaf = 1):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # для начала сортируем оба вектора: и признак, и таргет
    sorted_idx = np.argsort(feature_vector)
    feature_vector = feature_vector[sorted_idx]
    target_vector = target_vector[sorted_idx]
    
    # тут получаем пороги
    thresholds = (np.unique(feature_vector)[1:] + np.unique(feature_vector)[:-1]) / 2
    
    # тут немного стремный код, чтобы найти именно последние индексы повторяющихся элементов в массиве
    last_indices = np.unique(feature_vector[::-1], return_index=True)[1]
    last_indices = np.sort(len(feature_vector) - 1 - last_indices)
    
    # то же самое, только теперь разворачиваем массивы, т.е. то что сверху нужно будет для левой вершины,
    # а теперь для правой
    x_rev = feature_vector[::-1]
    y_rev = target_vector[::-1]
    last_indices_rev = np.unique(x_rev[::-1], return_index=True)[1]
    last_indices_rev = np.sort(len(y_rev) - 1 - last_indices_rev)
    
    # теперь считаем куммулятивную сумму для у, чтобы избежать циклов и посчитать долю у для каждого класса
    # Нас будут интересовать значения именно для last_indices
    p_left_1 = (np.cumsum(target_vector)[last_indices] / (np.arange(len(target_vector)) + 1)[last_indices])[:-1]
    p_left_0 = 1 - p_left_1
    
    p_right_1 = (np.cumsum(y_rev)[last_indices_rev] / (np.arange(len(y_rev)) + 1)[last_indices_rev])[:-1][::-1]
    p_right_0 = 1 - p_right_1
    
    
    # считаем доли объектов которые попали влево и вправо
    left = (np.arange(len(target_vector)) + 1)[last_indices][:-1] / len(target_vector)
    right = (len(target_vector) - (np.arange(len(target_vector)) + 1)[last_indices][:-1])  / len(target_vector)
    # ну и финалочка
    H_left = 1 - (p_left_0 ** 2) - (p_left_1 ** 2)
    H_right = 1 - (p_right_0 ** 2) - (p_right_1 ** 2)
    
    Q = - left * H_left - right * H_right
    
    # заменяем значения критерия Джини на сильно отрицательные для тех порогов, которые не подходят по min_samples_leaf
    left_obs = (np.arange(len(target_vector)) + 1)[last_indices][:-1]
    right_obs = (len(target_vector) - (np.arange(len(target_vector)) + 1)[last_indices][:-1])
    mask = np.logical_or((left_obs < min_samples_leaf), (right_obs < min_samples_leaf))
    if np.sum(mask) > 0:
        Q[mask] = - 10 ** 6
        
    threshold_best = thresholds[np.argmax(Q)]
    return thresholds, Q, threshold_best, Q.max()
    

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        
        self._min_samples_leaf = min_samples_leaf
        self.depth = 0

    def _fit_node(self, sub_X, sub_y, node):
        sub_X = np.array(sub_X)
        sub_y = np.array(sub_y)
        
        # проверяем условие для max_depth
        if (self._max_depth != None) and (self.depth >= self._max_depth):
            node["type"] = "terminal"
            node["class"] = np.argmax(np.bincount(sub_y)) 
            return
        # проверяем условие для min_samples_split
        if (self._min_samples_split != None) and (len(sub_y) < self._min_samples_split):
            node["type"] = "terminal"
            node["class"] = np.argmax(np.bincount(sub_y))
            return
        
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(np.unique(feature_vector)) == 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y, self._min_samples_leaf)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        self.depth += 1 # повышаем глубину на 1 для каждой итерации
        
        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['class']
        
        split = node['feature_split']
        if self._feature_types[split] == "real":
            threshold = node["threshold"]
            
            if x[split] < threshold:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])
        else:
            cats = node['categories_split']
            
            if x[split] in cats:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])
        

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

from sklearn.linear_model import LinearRegression
def find_best_split_reg(feature_vector, target_vector, min_samples_leaf = 1):
    # для начала сортируем оба вектора: и признак, и таргет
    sorted_idx = np.argsort(feature_vector)
    feature_vector = feature_vector[sorted_idx]
    target_vector = target_vector[sorted_idx]
    
    # перебираем пороги по квантилям
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresholds = np.quantile(feature_vector, quantiles)
    
    # дальше все просто: для каждого порога считаем лосс, записываем, находим минимальный
    losses = []
    for threshold in thresholds:
        features_left = feature_vector[feature_vector < threshold]
        y_left = target_vector[feature_vector < threshold]
        features_right = feature_vector[feature_vector >= threshold]
        y_right = target_vector[feature_vector >= threshold]
        n_left = len(features_left)
        n_right = len(features_right)
        n = len(feature_vector)
        
        model_1 = LinearRegression()
        model_2 = LinearRegression()
        model_1.fit(features_left.reshape(-1, 1), y_left)
        model_2.fit(features_right.reshape(-1, 1), y_right)
        pred_1 = model_1.predict(features_left.reshape(-1, 1))
        pred_2 = model_2.predict(features_right.reshape(-1, 1))
        mse_1 = ((pred_1 - y_left) ** 2).mean()
        mse_2 = ((pred_2 - y_right) ** 2).mean()
        
        if (n_left < min_samples_leaf) or (n_right < min_samples_leaf):
            loss = 10 ** 6
        else:
            loss = (n_left / n) * mse_1 + (n_right / n) * mse_2
        losses.append(loss)
        
    threshold_best = thresholds[np.argmin(losses)]
    
    return thresholds, losses, threshold_best, np.array(losses).min()

class LinearRegressionTree():
    def __init__(self, feature_types, base_model_type=None, max_depth=None, 
                 min_samples_split=None, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")
        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        
        self._min_samples_leaf = min_samples_leaf
        self.depth = 0

    def _fit_node(self, sub_X, sub_y, node):
        sub_X = np.array(sub_X)
        sub_y = np.array(sub_y)
        
        # проверяем условие для max_depth
        if (self._max_depth != None) and (self.depth >= self._max_depth):
            node["type"] = "terminal"
            model = LinearRegression()
            model.fit(sub_X, sub_y)
            node["model"] = model
            return
        # проверяем условие для min_samples_split
        if (self._min_samples_split != None) and (len(sub_y) < self._min_samples_split):
            node["type"] = "terminal"
            model = LinearRegression()
            model.fit(sub_X, sub_y)
            node["model"] = model
            return

        feature_best, threshold_best, loss_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(np.unique(feature_vector)) == 1:
                continue

            _, _, threshold, loss = find_best_split_reg(feature_vector, sub_y, self._min_samples_leaf)
            if loss_best is None or loss < loss_best:
                feature_best = feature
                loss_best = loss
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            model = LinearRegression()
            model.fit(sub_X, sub_y)
            node["model"] = model
            return
        
        self.depth += 1 # повышаем глубину на 1 для каждой итерации
        
        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            model = node['model']
            pred = model.predict(x.reshape(1, -1))
            return pred
        
        split = node['feature_split']
        if self._feature_types[split] == "real":
            threshold = node["threshold"]
            
            if x[split] < threshold:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])
        else:
            cats = node['categories_split']
            
            if x[split] in cats:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])
        

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
