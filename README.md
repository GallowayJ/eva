# Generalized extreme value utils

This package contains code to scrape, parse, tidy and visualise data from the Met Office’s historical station data.
In general, classes create dictionaries where the key is the station name and the value is a `pandas.DataFrame` of measurements.
Most methods update these DataFrames inplace, however in cases where methods perform a task where is not correct or improve format, a similar object to arg will be returned.
For example, in the case of creating a dictionary where values of station extreme values (annual maxima), instead of overwrite a instance attribute, a new dictionary will be returned.

For plotting utils, in general data in addition to a `matplotlib.pyplot` fig and ax(es) objects are require and the fig and ax(es) will always be returned allowing for further customization.

Documentation is available [here](https://extreme-value-anaylsis-utils.readthedocs.io/en/latest/).
