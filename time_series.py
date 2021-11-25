import pandas as pd
import numpy as np
import math
from tsfresh import extract_features, select_features
import tree_code as tc

def prepare_manufacturing_file(df, id):
    df = df.groupby(id).agg({list, "last"})
    df.columns = [' '.join(col).replace(" ", "") for col in df.columns]
    df['data_diameterlist'] = df['data_diameterlist'].apply(np.array)
    X = []
    values = df[['data_diameterlist']].copy()
    for v in values['data_diameterlist']:
        v = v[~np.isnan(v)]
        X.append(v)
    df['data_diameterlist'] = X
    df[id] = df.index
    df = df.dropna()
    colnames_numerics_only = df.select_dtypes(include=np.number).columns.tolist()
    return df, colnames_numerics_only

def prepare_dataset_running(df, id):
    df = df.groupby(id).agg({list, "last"})
    df = df.dropna(axis=1)
    df.columns = [' '.join(col).replace(" ", "") for col in df.columns]
    df['temperaturelist'] = df['temperaturelist'].apply(np.array)
    X = []
    values = df[['temperaturelist']].copy()
    for v in values['temperaturelist']:
        v = v[~np.isnan(v)]
        X.append(v)
    df['temperaturelist'] = X
    colnames_numerics_only = df.select_dtypes(include=np.number).columns.tolist()
    return df, colnames_numerics_only

def generate_interval_features(df, n, variable_interest):
    new_names = [variable_interest+str(inter) for inter in range(1, n+1)]
    def split(x, n):
        arrayss = np.array_split(x, n)
        return arrayss

    df[new_names] = (df.apply(lambda row: split(row[variable_interest], n),axis=1, result_type="expand"))
    for name in new_names:
        df[name+"_min"] = df.apply(lambda row: min(row[name], default=row[name]), axis=1)
        df[name + "_max"] = df.apply(lambda row: max(row[name], default=row[name]), axis=1)
        df[name + "_mean"] = df.apply(lambda row: np.mean(row[name]), axis=1)
        df[name + "_wthavg"] = df.apply(lambda row: np.average(row[name]), axis=1)
        df[name + "_sum"] = df.apply(lambda row: sum(row[name]), axis=1)
        df[name + "_std"] = df.apply(lambda row: np.std(row[name]), axis=1)
        try:
            df[name + "_slope"] = df.apply(lambda row: math.sqrt(row[name + "_max"] - row[name + "_min"])**2 + n**2, axis=1)/n #wurzel ((max-min)2 + n2)
            df[name + "_percentchange"] = (df[name + "_max"]- df[name + "_min"])/ df[name + "_min"] #(max-min)/min
        except:
            pass
    return df

def generate_global_features(df, y_var, id, interest):
    #https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html
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
        expression = interest_variable+ "list.count(" + str(data) + ")>=" + str(counter)
        new_name = str(expression)
        expression = "sum(i >=" + str(data) + "for i in list(row."+ interest_variable +"list))" + ">=" + str(counter)
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

def get_candidate_variables(df,id):
    df = df.sort_values(by=[id, 'time:timestamp'])
    variable_names = list(df)
    temp_var = []
    cand = []
    reoccuring_variables = []
    uuid = df.iloc[0,0]
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
        subsetDataFrame = df[df[id]==uuid]
        values = ((subsetDataFrame[variable_interest].to_numpy()))
        values = [v for v in values if not math.isnan(v)]
        result = list(subsetDataFrame[result_column])
        if variable_result in result:
            result = "NOK"
            array_nok.append(values)
            uuids_complete.append(uuid)
            if (len(values)>0):
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

def pipeline(use_case, df, id, variable_result, result_column, variable_interest):
    candidates, array_ok, array_nok, uuids_complete = sort_array_ok_nok(df, id, variable_result, variable_interest,
                                                                        result_column)
    candidate_vars = get_candidate_variables(df, id)
    print("Reoccuring variables/Time Series Candidates: ", list(candidate_vars))

    if use_case == "manufacturing":
        df_newFeatures = df[[id, variable_interest]]
        df_newFeatures = df_newFeatures.dropna()
        y_var = df[[id, result_column]].groupby(id).agg('last').dropna().reset_index()
        y_var = y_var[y_var[id].isin(df_newFeatures[id].values)]
        y_var = y_var[result_column].to_numpy()
        df, num_cols = prepare_manufacturing_file(df, id)
        df = df.reset_index(drop=True)

    else:
        df_newFeatures = df.select_dtypes(include=['number'])
        df, num_cols = prepare_dataset_running(df, id)
        df = df.dropna(axis=1)
        y_var = df[result_column+"last"].to_numpy()

    result_column = result_column + "last"

    #Interval-Based:
    interval = [2,5,10]
    max_i = interval[0]
    accuracy_baseline = 0
    used_features = []
    try:
        for i in interval:
            df_new = generate_interval_features(df, i, variable_interest + "list")
            df_new = df_new.dropna()
            var_interval = df_new.select_dtypes(include=np.number).columns.tolist()
            var_interval = [x for x in var_interval if x != id]
            accuracy, used_features = tc.learn_tree(df_new, result_column, var_interval, variable_result)
            if accuracy > accuracy_baseline:
                accuracy_baseline = accuracy
                max_i = i
                max_features = used_features
    except:
        pass
    # df = generate_interval_features(df, max_i, variable_interest + "list")

    df = generate_interval_features(df, max_i, variable_interest + "list")
    df = df_new.dropna()
    #var_interval = df_new.select_dtypes(include=np.number).columns.tolist()
    #var_interval = [x for x in var_interval if x != id]
    var_interval = max_features
    #accuracy, used_features = tc.learn_tree(df_new, result_column, var_interval, variable_result)

    #pattern based
    candidate_thresholds = get_distribution(array_ok, array_nok)
    df, var_pattern = create_latent_variables(candidate_thresholds, df, variable_interest)
    for var in var_pattern:
         var_interval.append(var)
    tc.learn_tree(df, result_column, var_interval, variable_result)

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
    #tc.learn_tree(df, result_column, num_cols, False)

    #combined
    for v in var_interval:
          num_cols.append(v)
    num_cols = [x for x in num_cols if x != id]
    tc.learn_tree(df, result_column, num_cols, variable_result, True)

use_case = "manufacturing"

if use_case == "manufacturing":
    file = "data/manufacturing.csv"
    id = "casename"
    variable_result = "nok"
    result_column = "case:data_success"
    variable_interest = "data_diameter"
    df = pd.read_csv(file)
    df = df.rename(columns={'sub_concept' : 'subname', 'case:concept:name' : 'casename'})
    subuuids = dict()
    for index, row in df.iterrows():
        #print(row.casename, row.subname)
        if not math.isnan(row.subname):
            if row.casename in subuuids:
                if type(subuuids[row.casename]) != list:
                    temp = []
                    temp.append(subuuids[row.casename])
                    temp.append(row.subname)
                    subuuids[row.casename] = temp
                    #print(temp)
                else:
                    temp = subuuids[row.casename]
                    temp.append(row.subname)
                    subuuids[row.casename] = temp
                    #print(subuuids[row.uuid])
            else:
                subuuids[row.casename] = row.subname
    for key, value in subuuids.items():
        subuuids[key] = list(set(value))
    i = 0
    uuids = dict()
    for key, value in subuuids.items():
        for key1, subkeys in subuuids.items():
            for subkey in subkeys:
                #print(key, int(v))
                if key == int(subkey):
                    subsub = subuuids[subkey][0]
                    uuids[subkey] = key1
                    uuids[subsub] = key1
                    i = i +1
    df = df.replace({'casename': uuids})
    df = df.drop(columns="subname")
    df = df.drop(columns="sub_uuid")
    pipeline(use_case, df, id, variable_result, result_column, variable_interest)

elif use_case == "running":
    file = 'data/running.csv'
    id = "uuid"
    result_column = 'event'
    variable_result = 'Discard Goods'
    variable_interest = "temperature"
    df = pd.read_csv(file)
    df = df.rename(columns={"timestamp": "time:timestamp"})
    pipeline(use_case, df, id, variable_result, result_column, variable_interest)

