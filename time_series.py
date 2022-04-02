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

import pandas as pd
import numpy as np
import math
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import sys
from tsfresh import extract_features, select_features
import tree_code as tc


def prepare_dataset(df, id, variable_interest):
    df = df.groupby(id).agg({list, "last"})
    df.columns = [' '.join(col).replace(" ", "") for col in df.columns]
    df[variable_interest + 'list'] = df[variable_interest + 'list'].apply(np.array)
    X = []
    values = df[[variable_interest + 'list']].copy()
    for v in values[variable_interest + 'list']:
        v = v[~np.isnan(v)]
        X.append(v)
    df[variable_interest + 'list'] = X
    df = df.dropna()
    colnames_numerics_only = df.select_dtypes(include=np.number).columns.tolist()
    return df, colnames_numerics_only


def generate_interval_features(df, n, variable_interest):
    new_names = [variable_interest + str(inter) for inter in range(1, n + 1)]

    def split(x, n):
        arrayss = np.array_split(x, n)
        return arrayss

    df[new_names] = (df.apply(lambda row: split(row[variable_interest], n), axis=1, result_type="expand"))
    for name in new_names:
        df[name + "_min"] = df.apply(lambda row: min(row[name], default=row[name]), axis=1)
        df[name + "_max"] = df.apply(lambda row: max(row[name], default=row[name]), axis=1)
        df[name + "_mean"] = df.apply(lambda row: np.mean(row[name]), axis=1)
        df[name + "_wthavg"] = df.apply(lambda row: np.average(row[name]), axis=1)
        df[name + "_sum"] = df.apply(lambda row: sum(row[name]), axis=1)
        df[name + "_std"] = df.apply(lambda row: np.std(row[name]), axis=1)
        try:
            df[name + "_slope"] = df.apply(lambda row: math.sqrt(row[name + "_max"] - row[name + "_min"]) ** 2 + n ** 2,
                                           axis=1) / n  # wurzel ((max-min)2 + n2)
            df[name + "_percentchange"] = (df[name + "_max"] - df[name + "_min"]) / df[name + "_min"]  # (max-min)/min
        except:
            pass
    return df


def generate_global_features(df, y_var, id, interest):
    # https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html
    extracted_features = extract_features(df, column_id=id, column_value=interest)
    extracted_features = extracted_features.dropna(axis=1)
    try:
        features_filtered = select_features(extracted_features, y_var)
    except:
        features_filtered = extracted_features

    features_filtered[id] = features_filtered.index
    features_filtered.reset_index()
    return features_filtered


def create_latent_variables(candidate_thresholds, df, interest_variable):
    new_names = []
    for c in candidate_thresholds:
        data = c[0]
        counter = c[1]
        expression = interest_variable + "list.count(" + str(data) + ")>=" + str(counter)
        new_name = str(expression)
        expression = "sum(i >=" + str(data) + "for i in list(row." + interest_variable + "list))" + ">=" + str(counter)
        df[new_name] = df.apply(lambda row: (eval(expression)), axis=1)
        new_names.append(new_name)
    return df, new_names


def get_distribution(array_ok, array_nok):
    frequencies = []
    for ok in array_ok:
        (unique, counts) = np.unique(ok, return_counts=True)
        f = np.asarray((unique, counts)).T
        frequencies.append(f)
    frequencies_average_ok = dict()
    for f in frequencies:
        for value, counter in f:
            if (value, counter) not in frequencies_average_ok:
                frequencies_average_ok[value, counter] = 1
            if (value, counter) in frequencies_average_ok:
                frequencies_average_ok[value, counter] = frequencies_average_ok[value, counter] + 1

    frequencies = []
    for nok in array_nok:
        (unique, counts) = np.unique(nok, return_counts=True)
        f = np.asarray((unique, counts)).T
        frequencies.append(f)
    frequencies_average_nok = dict()
    for f in frequencies:
        for value, counter in f:
            if (value, counter) not in frequencies_average_nok:
                frequencies_average_nok[value, counter] = 1
            if (value, counter) in frequencies_average_nok:
                frequencies_average_nok[value, counter] = frequencies_average_nok[value, counter] + 1

    diff = []
    for f in frequencies_average_nok.keys():
        if f not in frequencies_average_ok:
            diff.append(f)
    return diff


def get_candidate_variables(df, id):
    df = df.sort_values(by=[id, 'time:timestamp'])
    variable_names = list(df)
    temp_var = []
    cand = []
    reoccuring_variables = []
    uuid = df.iloc[0, 0]
    for column, row in df.iterrows():
        for var in variable_names:
            if var == id:
                if row[var] == uuid:
                    continue
                else:
                    uuid = row[var]
                    reoccuring_variables.extend(cand)
                    temp_var = []
            else:
                if not (isinstance(row[var], (int, float, np.int64, bool))):
                    continue
                else:
                    if var in temp_var:
                        cand.append(var)
                    else:
                        temp_var.append(var)
    reoccuring_variables = set(reoccuring_variables)
    constant_variables = [c for c in reoccuring_variables if len(set(df[c])) == 1]
    candidate_var = [c for c in reoccuring_variables if c not in constant_variables]
    return candidate_var


def sort_array_ok_nok(df, id, variable_result, variable_interest, result_column):
    candidates = dict()
    uuids = set(df[id])
    uuids_complete = []
    array_ok = []
    array_nok = []
    for uuid in uuids:
        subsetDataFrame = df[df[id] == uuid]
        values = ((subsetDataFrame[variable_interest].to_numpy()))
        values = [v for v in values if not math.isnan(v)]
        result = list(subsetDataFrame[result_column])
        if variable_result in result:
            result = "NOK"
            array_nok.append(values)
            uuids_complete.append(uuid)
            if (len(values) > 0):
                candidates[uuid] = [result, values]
        else:
            result = "OK"
            array_ok.append(values)
            uuids_complete.append(uuid)
            if (len(values) > 0):
                candidates[uuid] = [result, values]
    print("Array NOK: ", len(array_nok))
    print("Array OK: ", len(array_ok))
    return candidates, array_ok, array_nok, uuids_complete


def pipeline(use_case, df, id, variable_result,results,result_column, variable_interest=None, interval=None):
    candidates, array_ok, array_nok, uuids_complete = sort_array_ok_nok(df, id, variable_result, variable_interest,
                                                                        result_column)
    candidate_vars = get_candidate_variables(df, id)
    candidate_vars = [x for x in candidate_vars if x != result_column]
    print("Reoccuring variables/Time Series Candidates: ", (candidate_vars))
    num_cols_all = []
    df_og = df
    df_reset = df
    result_column_og = result_column


    for c in candidate_vars:
        df = df_reset
        variable_interest = c
        if use_case == "manufacturing":
            df_newFeatures = df[[id, variable_interest]]
            df_newFeatures = df_newFeatures.dropna()
            y_var = df[[id, result_column_og]].groupby(id).agg('last').dropna().reset_index()
            y_var = y_var[y_var[id].isin(df_newFeatures[id].values)]
            y_var = y_var[result_column_og].to_numpy()
            df, num_cols = prepare_dataset(df, id, variable_interest)
            df[id] = df.index
            df = df.reset_index(drop=True)

        else:
            df_newFeatures = df.select_dtypes(include=['number'])
            df, num_cols = prepare_dataset(df, id, variable_interest)
            df = df.dropna(axis=1)
            y_var = df[result_column_og + "last"].to_numpy()

        result_column = result_column_og + "last"
        max_accuracy = 0

        # Interval-Based:
        if not interval:
            interval = [2, 5, 10]
        else:
            interval = interval
        max_i = interval[0]
        accuracy_baseline = 0
        max_features = []
        try:
            for i in interval:
                df_new = generate_interval_features(df, i, variable_interest + "list")
                df_new = df_new.dropna()
                var_interval = df_new.select_dtypes(include=np.number).columns.tolist()
                var_interval = [x for x in var_interval if x != id]
                accuracy, used_features = tc.learn_tree(df_new, result_column, var_interval, variable_result, False)
                if accuracy > accuracy_baseline:
                    accuracy_baseline = accuracy
                    max_i = i
                    max_features = used_features
        except:
            pass
        df = generate_interval_features(df, max_i, variable_interest + "list")
        df = df.dropna()
        if accuracy_baseline > max_accuracy:
            max_accuracy = accuracy_baseline
            var_interval = max_features
        else:
            var_interval = []
        print("Calculated interval-based features...")

        # pattern based
        candidate_thresholds = get_distribution(array_ok, array_nok)
        df, var_pattern = create_latent_variables(candidate_thresholds, df, variable_interest)
        accuracy, var_pattern = tc.learn_tree(df, result_column, var_pattern, variable_result)
        if accuracy > max_accuracy:
            for var in var_pattern:
                var_interval.append(var)
        print("Calculated pattern-based features...")

        #global features
        df_newFeatures = df_newFeatures.dropna().reset_index()
        global_features = generate_global_features(df_newFeatures, y_var, id, variable_interest)
        df = pd.merge(df, global_features, on=id)
        to_drop = []
        for d in df.columns:
            if np.inf in df[d].values:
                to_drop.append(d)
        df = df.drop(columns=to_drop)
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        accuracy, num_cols = tc.learn_tree(df, result_column, num_cols, variable_result)
        print("Calculated global features...")

        # combined
        for v in var_interval:
            num_cols.append(v)
        num_cols = [x for x in num_cols if x != id]
        if len(candidate_vars) == 1:
            df_og = df
            num_cols_all = num_cols
        else:
            df_og = pd.merge(df, df_og, on=id, how="outer", suffixes=('', '_y'))
            num_cols_all.extend(num_cols)

    tc.learn_tree(df_og, result_column, num_cols_all, variable_result, results, True)

if __name__ == '__main__' :

    try:
        use_case = sys.argv[1]
    except:
        use_case = "running"

    #preprocessing happens here
    if use_case == "manufacturing":
        file = "data/manufacturing.csv"
        id = "casename"
        results = ['nok', 'ok']
        variable_result = "nok"
        result_column = "case:data_success"
        variable_interest = "data_diameter"
        df = pd.read_csv(file)
        df = df.rename(columns={'sub_concept': 'subname', 'case:concept:name': 'casename'})
        subuuids = dict()
        for index, row in df.iterrows():
            if not math.isnan(row.subname):
                if row.casename in subuuids:
                    if type(subuuids[row.casename]) != list:
                        temp = []
                        temp.append(subuuids[row.casename])
                        temp.append(row.subname)
                        subuuids[row.casename] = temp
                    else:
                        temp = subuuids[row.casename]
                        temp.append(row.subname)
                        subuuids[row.casename] = temp
                else:
                    subuuids[row.casename] = row.subname
        for key, value in subuuids.items():
            subuuids[key] = list(set(value))
        i = 0
        uuids = dict()
        for key, value in subuuids.items():
            for key1, subkeys in subuuids.items():
                for subkey in subkeys:
                    if key == int(subkey):
                        subsub = subuuids[subkey][0]
                        uuids[subkey] = key1
                        uuids[subsub] = key1
                        i = i + 1
        df = df.replace({'casename': uuids})
        df = df.drop(columns="subname")
        df = df.drop(columns="sub_uuid")
        pipeline(use_case, df, id, variable_result, results, result_column, variable_interest)

    else:
        use_case = "running"
        file = 'data/running.csv'
        id = "uuid"
        results = ['Discard Goods', 'Transfer Goods']
        result_column = 'event'
        variable_result = 'Discard Goods'
        variable_interest = "temperature"
        df = pd.read_csv(file)
        df = df.rename(columns={"timestamp": "time:timestamp"})
        pipeline(use_case, df, id, variable_result, results, result_column, variable_interest)

