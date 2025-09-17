.. GEV utils documentation master file, created by
   sphinx-quickstart on Tue Sep  9 16:29:20 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Generalized Extreme Values (GEV) utils documentation
==============================================

This package contains code to scrape, parse, tidy and visualise data from the `Met Office's historical station data`_.
In general, classes create dictionaries where the key is the station name and the value is a pandas.DataFrame of measurements.
Most methods update these DataFrames inplace, however in cases where methods perform a task where is not correct or improve format, a
similar object to arg will be returned. For example, in the case of creating a dictionary where values of station extreme values (annual maxima),
instead of overwrite a instance attribute, a new dictionary will be returned.

For plotting utils, in general data in addition to a matplotlib.pyplot fig and ax(es) objects are require and the fig and ax(es) will always be
returned allowing for further customization.


.. _Met Office's historical station data: https://www.metoffice.gov.uk/research/climate/maps-and-data/historic-station-data

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   installation
   gev_utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

