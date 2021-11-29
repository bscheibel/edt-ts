# edt-ts

Requirements:
* OS: Fedora 34 
* Python: version 3.9.6 
* Python packages: 
    * pandas==1.2.0
    * numpy==1.20.1
    * scikit_learn==0.24.0
    * tsfresh==0.18.0

These packages can be installed using pip, the version is important (especially for tsfresh). 

The folder 'data' includes csv files for the running example as well as the manufacturing use case.
The manufacturing use case data was originally in yaml form, converted to XES and then to csv. 
The full results, including the baseline results, can be seen in result_{running_example, manufacturing}.txt

To start the script in terminal: python time_series.py {running, manufacturing}.
Per default, the running example use case is started. 

OR run using pipenv, which creates a virtual environment with all needed packages:
* python 3.9 needed
* install pipenv via pip
* run "pipenv install"
* run "pipenv run python time_series.py {use_case}"

Parameters that have to be set: 
* use_case: if one of the existing use cases is to be reproduced, just enter the use case name e.g. running, manufacturing 

For a new dataset: 
* df: the dataframe has to be provided if a new dataset should be tried 
* id: the identifier of the instances e.g. uuid 
* result_column: whats the name of the column that specifies the result 
* variable_result: if more than two categories exist, which category is of interest 
* results: all possible result classes

Optionally: 
* interval: if the intervals are to be set manually
* variable_interest: if not given, possible time series variables are discovered and all are included in the process

To try with new dataset, these variables have to be defined as part of the preprocessing step. In addition, the dataset may has to be transformed. Look at preprocessing code for the manufacturing use case for an example.
