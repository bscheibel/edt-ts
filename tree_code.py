# Copyright (C) 2021  Beate Scheibel
# This file is part of edt-ts.
#
# edt-ts is free software: you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# edt-ts is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# edt-ts.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.ensemble import ExtraTreesClassifier
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.tree import export_text
import re
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

random_seed = 2
RANDOMSEED = 42
def get_rules(model, features, results, result):
    # adapted from: https://mljar.com/blog/extract-rules-decision-tree/
    tree_ = model.tree_
    feature_name = [
        features[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    paths = []
    path = []
    def recurse(node, path, paths):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    rules = []
    for path in paths:
        rule = "IF "
        for p in path[:-1]:
            if rule != "IF ":
                rule += " AND "
            p = re.sub("> 0.5", "== TRUE", p)
            p = re.sub("< 0.5", "== FALSE", p)
            rule += str(p)
        rule += " THEN "
        classes = path[-1][0][0]
        l = np.argmax(classes)
        if results[l] != result:
            continue
        else:
            rule += f"class: {result} "
            rules += [rule]
    return rules

def learn_tree(df, result_column, names, result, results=None, final=False):
    y_var = df[result_column].values
    X_var = df[names]
    features = np.array(list(X_var))

    clf = ExtraTreesClassifier(n_estimators=50, random_state=RANDOMSEED)
    clf = clf.fit(X_var, y_var)
    model = SelectFromModel(clf, prefit=True, max_features=5)
    features = features[model.get_support()]
    X_var = X_var[features]
    features = list(X_var)
    X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.20, shuffle=False, stratify=None, random_state=RANDOMSEED)
    clf = tree(random_state=RANDOMSEED)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = tree(random_state=1, ccp_alpha=ccp_alpha, splitter="best")
        clf.fit(X_train, y_train)
        clfs.append(clf)
    if len(clfs) > 1:
        clfs = clfs[:-1]

    score_ = [precision_score(y_test, clf.fit(X_train, y_train).predict(X_test), average="weighted") for clf in clfs]
    #score_ = [clf.score(X_test, y_test) for clf in clfs]

    index_max = np.argmax(score_)
    if not final:
        model = clfs[-1]
    else:
        model = clfs[-1]
    model = model.fit(X_train,y_train)

    pred_model = model.predict(X_test)
    n_nodes = model.tree_.node_count
    max_depth = model.tree_.max_depth
    accuracy = accuracy_score(y_test, pred_model)
    precision = precision_score(y_test, pred_model, average=None)
    used_features = []
    i = 0
    try:
        for f in model.feature_importances_:
            if f > 0:
                used_features.append(features[i])
            i = i + 1
    except:
        pass

    if final:
        print("Number of nodes total: ", n_nodes)
        print("Max depth: ", max_depth)
        print(('Accuracy of the model is {:.0%}'.format(accuracy)))
        print("Precision: ", precision)
        print("Used features: ", used_features)
        tree_rules = export_text(model, feature_names=features)
        print(tree_rules)
        rules = (get_rules(model, features, results, result))
        for r in rules:
            print(r)
    return accuracy, used_features
