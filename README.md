# edt-ts

OS: Fedora 34 \

Python: version 3.9 \
 
Python modules: \
* pandas \
* numpy \
* tslearn \
* sklearn \
* tsfresh \

These packages can be installed using pip, the version number is important (especially for tsfresh). \

The folder 'data' includes csv files for the running example as well as the manufacturing use case.
The manufacturing use case data was originally in yaml form, converted to XES and then to csv. 

To start the script in terminal: python time_series.py {running, manufacturing}.

To try with additional data the preprocessing step has to be adapted to fit the new data to the required format. 
See the functions "preprocess_running" and "preprocess_manufacturing" on what is expected.