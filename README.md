# edt-ts

Requirements:
* OS: Fedora 34 
* Python: version 3.9 
*Python modules: 
    * pandas 
    * numpy 
    * tslearn 
    * sklearn
    * tsfresh 

These packages can be installed using pip, the version is important (especially for tsfresh). 

The folder 'data' includes csv files for the running example as well as the manufacturing use case.
The manufacturing use case data was originally in yaml form, converted to XES and then to csv. 
The full results, including the baseline results, can be seen in result_{running, manufacturing}.txt

To start the script in terminal: python time_series.py {running, manufacturing}.
Per default, the running example use case is started. 

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


To try with additional data the dataset might has to be prepared. Look at preprocessing code for an example. The dataset for the running example can be directly used, only the input arguments have to be defined.