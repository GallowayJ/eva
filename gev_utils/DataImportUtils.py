from io import StringIO
import os.path, os.mkdir
import time
import re

from bs4 import BeautifulSoup
import geopandas as gpd
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import requests


class WebScraper:
    """Scrapes met station data and parses data into a usable format.

    Attributes:
        self.base_url (str): The base url to be scraped
        self.station_links (dict): A dictionary where keys are station
            names and values are links to station data.
        self.station_dict (dict): A dictionary where keys are station
            names and values are a list of data and metadata.
        self.nonstandard_list (list): A list of stations where the
            format of the metadata does not permit automatic parsing.
    """
    def __init__(self, base_url):
        """Construct an instance of WebScraper.

        Args:
            base_url (str) : The url of the met station website.

        Returns:
            An instance of webscraper. NB WebScraper.get_data() must be
            called for scraping to take place.
        """
        self.base_url = base_url
        self.station_links = {}
        self.station_dict = {}
        # List of stations in an unparseable format.
        self.nonstandard_list = []

    def get_stations_and_links(self):
        """Scrapes base_url for met station names and met station links.
        Updates self.station_links and creates a dictionary where keys
        are met station names and values are met station links.

        Returns:
            None
        """
        page = requests.get(self.base_url)
        soup = BeautifulSoup(page.content, features="html.parser")
        # skip first element (not a station)
        self.station_links = soup.select("option")[1:]

        # Create dictionary where keys are station names and values are
        # a list of length 1 containing the link to station historical
        # data.
        for each_tag in self.station_links:
            station_name = each_tag.text
            station_link = each_tag.attrs["value"]
            self.station_dict.update({station_name: [station_link]})

    def get_coordinates(self):
        """Attempts to extract coordinates from station metadata. If
        this fails, add the station name to self.nonstandard_list.

        Returns:
            None
        """
        for k, v in self.station_dict.items():
            try:
                lat, lon = [
                    float(coord)
                    for coord in re.findall(r"[-.\d]+", v[2][1].split(",")[1])
                ]
                self.station_dict[k].extend([lat, lon])
            except ValueError:
                lat, lon = [
                    float(coord)
                    for coord in re.findall(r"[-.\d]+", v[2][2].split(",")[1])
                ]
                print(f"{k} was relocated during the timeseries.")
                self.station_dict[k].extend([lat, lon])
                pass
            except IndexError:
                self.nonstandard_list.extend([k])

    def get_data(self):
        """Scrapes met station data (measurements) and metadata
        (coordinates, whether a station was relocated) from
        self.base_url. Modifies self.station_dict values to contain a
        list of the station's link, a pandas.DataFrame of station
        measurements and a list metadata.

        Returns:
            self.station_dict (dict): A dict where keys are station
            names and is a list of station data and metadata.
        """
        self.get_stations_and_links()
        # get station data, parse into pandas DataFrames and retrieve
        # station metadata
        for k, v in self.station_dict.items():
            r = requests.get(v[0])
            data = BeautifulSoup(r.content, features="html.parser")
            metadata = StringIO(data.text).readlines()[:5]
            df_cols = [
                "Year",
                "Month",
                "tMax",
                "tMin",
                "Air Frost",
                "Rainfall",
                "Sun hours",
            ]
            df = pd.read_fwf(
                StringIO(data.text),
                skiprows=7,
                sep=" ",
                na_values=["---", "--"],
                names=df_cols,
            )
            v.extend([df, metadata])
            time.sleep(0.2)  # pause between GET requests
        self.get_coordinates()
        return self.station_dict


class DataMunger:
    """Carries out data munging of self.data_dict. Drops unneeded data
    columns, removes header units, strips characters from numeric fields
    so that correct dtypes can be assigned, extracts station coordinates
    from station metadata, creates a datatime index and contains a
    method to return an extreme value dict.

        Attributes:
            self.data_dict (dict): A dictionary where keys are station
                names and values are lists of station data and metadata.
            self.station_coords_list (list): A list of latitude and
                longitude extracted from station metadata.

    """

    def __init__(self, data_dict):
        """Constructs an instance of DataMunger.

        Args:
            data_dict (dict): A dict created by WebScraper.get_data()

        Returns:
            An instance of DataMunger.
        """
        self.data_dict = data_dict
        self.station_coords_list = []

    def drop_unneeded_cols(self, targ_col=["Rainfall"]):
        """Drops all columns apart from 'Year', 'Month' and targ_column.

        Kwargs:
            targ_col (list): A list containing names (str) of columns
            to keep.

        Returns:
            self.data_dict (dict): Modified data_dict consisting of
                columns in targ_col and 'Year' and 'Month' columns.
        """
        for k, v in self.data_dict.items():
            df = v[1]
            cols_to_keep = ["Year", "Month"]
            cols_to_keep.extend(targ_col)
            self.data_dict[k][1] = df[cols_to_keep]
        return self.data_dict

    def remove_units_header(self, targ_str="mm", targ_col="Rainfall"):
        """Removes the first row if it has been used to specify units of
        measure.

        Kwargs:
            targ_str (str): The unit string to be removed.

        Returns:
            None
    """
        for k, v in self.data_dict.items():
            df = v[1]
            if df.loc[0, targ_col] == targ_str:
                self.data_dict[k][1] = df.iloc[1:].reset_index(drop=True)
                self.data_dict[k][1] = df.dropna(how="all")

    def strip_asterices(self):
        """Iterates through self.data_dict, DataFrames containing
        asterices are removed. Each DataFrame in self.data_dict is
        modified in place.

        Returns:
            None
        """
        for k, v in self.data_dict.items():
            df = v[1]
            self.data_dict[k][1] = df.replace(r"\*", "", regex=True)

    def get_station_coords(self):
        """Iterates through self.data_dict, latitude and longitude is
        extracted from metadata and appended to self.station_coords_list.

        Returns:
            None
        """
        for k, v in self.data_dict.items():
            lat, lon = v[3], v[4]
            self.station_coords_list.append([k, lat, lon])

    def assign_correct_dtypes(self):
        """Iterates through self.data_dict. Rows containing missing
        observations are dropped, and the following dtypes are
        assigned: Year (Int64), Month (Int64), Rainfall (float32).

        Returns:
            None
        """
        for k, v in self.data_dict.items():
            self.data_dict[k][1] = self.data_dict[k][1].dropna(how="any")
            self.data_dict[k][1] = self.data_dict[k][1].apply(
                pd.to_numeric, axis=1, errors="coerce"
            )
            self.data_dict[k][1] = self.data_dict[k][1].astype(
                {"Year": "Int64", "Month": "Int64", "Rainfall": "float32"}
            )

    def create_dt_index(self):
        """Iterates through self.data_dict, Year and Month columns are
        used to create a datetime intex which is assigned to a new
        column, 'Date' which is set to the index of the DataFrame. Year
        and Month columns are then dropped.

        Returns:
            None
        """
        for k, v in self.data_dict.items():
            v[1]["Date"] = pd.to_datetime(
                dict(year=v[1]["Year"], month=v[1]["Month"], day=1)
            ) + MonthEnd(0)
            self.data_dict[k][1] = v[1].set_index("Date", drop=True)
            cols_to_drop = ["Year", "Month"]
            self.data_dict[k][1] = self.data_dict[k][1].drop(cols_to_drop, axis=1)

    def get_extreme_dict(self, data_dict):
        """Iterates through a data_dict and resamples by year, returning
        the maxima. It returns a dictionary containing the resampled
        DataFrames.

        Args:
            data_dict (dict): A dictionary where the keys are station
                names and values are DataFrames. DataFrames must have a
                datetime index.

            Returns:
                maxima_dict (dict): A dictionary where the keys are
                    station names and the values are DataFrames
                    containing the annual maxima.
        """
        maxima_dict = data_dict.copy()
        for each_site, each_df in maxima_dict.items():
            maxima_dict[each_site] = each_df.groupby(each_df.index.year).max()
            idx_max_dates = each_df.groupby(each_df.index.year).idxmax()
            # this column contain datetime of when maximum occured in
            # each period
            maxima_dict[each_site] = maxima_dict[each_site].set_index(
                idx_max_dates["Rainfall"], drop=True
            )
            return maxima_dict

    def tidy_data(self):
        """Carries out data munging by calling self.drop_unneeded_cols,
        self.remove_units_header, self.strip_asterices,
        self.assign_correct_dtypes, self.create_dt_index,
        self.get_station_coords.

        Returns:
            None
        """
        self.drop_unneeded_cols()
        self.remove_units_header()
        self.strip_asterices()
        self.assign_correct_dtypes()
        self.create_dt_index()
        self.get_station_coords()

    def save_data(self, directory, **kwargs):
        """Iterates through self.data_dict and saves each DataFrame into
        'directory' as a .csv named by the dictionary key (station name).

        Args:
            directory (str): The name of the directory where .csvs
                should be saved.

        Kwargs:
            Keyword arguments to pass to pandas.DataFrame.to_csv().
        """
        os.mkdir('outputs')
        for k, v in self.data_dict.items():
            fp = os.path.join(directory, f"{k}.csv")
            v[1].to_csv(fp, **kwargs)


class GeoMunger:
    """Handles parsing, unifification and tidying of spatial data and
    supporting information (house price data and modelling results).

    Attributes:
        self.house_price (pandas.DataFrame): A pandas.DataFrame of
            average house prices.
        self.station_coords (pandas.DataFrame): A pandas.DataFrame of
            station latitude and longitude.
        self.metrics_df (pandas.DataFrame): A pandas.DataFrame of
            predictions, error bounds and supporting data.
        self.unified_gdf (geopandas.GeoDataFrame): A geopandas.GeoDataFrame
            containing statistical data and geospatial data.
        self.aoi_outline (geopandas.GeoDataFrame): A geopandas.GeoDataFrame
            of the outline of the area of interest (aoi).
    """

    def __init__(self, gdf_house_price, gdf_station_coords, aoi_outline=None):
        """Constructs an instance of GeoMunger.

        Args:
            gdf_house_price (GeoDataFrame): A GeoDataFrame containing
                average house prices for each region.
            gdf_station_coords (GeoDataFrame): A GeoDataFrame
                containing the coordinates (lat, lon) of each station.

        Kwargs:
            aoi_outline (GeoDataFrame): A GeoDataFrame delinating the
                outline of the study area. Used for visualisation.

            Returns:
                An instance of GeoMunger.
        """
        self.house_price = gdf_house_price
        self.station_coords = gdf_station_coords
        self.metrics_df = None
        self.unified_gdf = None
        self.aoi_outline = aoi_outline

    def extract_station_metrics(self, gev_dict):
        """Extracts mean predicted return intensity and the 95% HDI
        from an instance of gev_dict and the corresponding return periods.

        Args:
            gev_dict (dict): An instance of gev_dict generated by
                fitting extreme events to the generalized extreme value
                distribution using PyMC. NB gev_dict is generated by
                the accompaning Jupyter Notebook.

        Returns:
            station_metrics (pd.DataFrame): A pandas DataFrame
                containing only the information needed for carrying out
                analysis in the accompaning Jupyter noteobook.
        """
        station_metrics = pd.DataFrame(
            {
                "Return intensity": gev_dict["az_summary"]["mean"],
                "Lower": gev_dict["az_summary"]["hdi_2.5%"],
                "Upper": gev_dict["az_summary"]["hdi_97.5%"],
                "Return period": gev_dict["return periods"],
            }
        ).reset_index(drop=True)
        return station_metrics

    def get_return_period_results(self, station_name, df, return_period_yrs=20):
        """This method takes a pandas DataFrame containing the predicted
        return period, intensity and confidence intervals and returns a
        single row DataFrame containing this metric in addtion to adding
        a column with the site name.

        Args:
            station_name (str): The name of the met station.
            df (pandas.DataFrame): A pandas.DataFrame containing the
                results of the posterior preditive containing the
                return period, return intensity and HDI (95% is used in
                this example).

        Kwargs:
            return_period_yrs (int): The return period of interest,
                default 20.

        Returns:
            df (pandas.DataFrame): A pandas.DataFrame containing the
                return intensity, lower and upper intervals for
                `return_period_yrs`.
        """

        df = df.iloc[[(df["Return period"] - return_period_yrs).abs().argsort()[0]]]
        df.insert(0, "Station name", station_name)
        return df

    def get_metrics_df(self, gev_dict):
        """Updates self.metric_df with mean prediction and lower and
        upper bound of the HDI (by default 95%) by extracting this
        information from `gev_dict`.

        Args:
            gev_dict (dict): A dict containing results modelling results
                from the accompaning Jupyter Notebook.

        Returns:
            None
        """

        metrics_dict = {k: self.extract_station_metrics(v) for k, v in gev_dict.items()}
        metrics_dict = {
            k: self.get_return_period_results(k, v) for k, v in metrics_dict.items()
        }
        self.metrics_df = pd.concat(metrics_dict, ignore_index=True)

    def add_metrics_to_stations(self):
        """Merges `self.station_coords` and `self.metrics_df` on station
        name with the result set to `self.metrics_df`.

        Returns:
            None
        """
        self.station_coords = self.station_coords.merge(
            self.metrics_df, on="Station name"
        )

    def convert_crs(self, crs="EPSG:27700"):
        """Converts the crs of `self.house_price` and
        `self.station_coords` to `crs`.

        Kwargs:
            crs (str): A valid crs string to be passed to
                geopandas.GeoDataFrame.to_crs(), default `EPSG:27700`,
                (recommended for the UK).

        Returns:
            None
        """
        self.house_price = self.house_price.to_crs(crs)
        self.station_coords = self.station_coords.to_crs(crs)

    def join_house_prices(self, cols_to_keep=None):
        """Joins `self.house_price` and `self.station_coords` using
        geopandas.sjoin_nearest() and sets the result to
        `self.unified_gdf`. The result is that met station will be
        assigned to their geographically nearest house price region.
        Also drops unneeded columns, a list of column names can be
        specified using `cols_to_keep` to change which columns are kept.

        Kwargs:
            cols_to_keep (list-like): A list of columns names to be kept
                in the resulting GeoDataFrame.

        Returns:
            None
        """
        self.unified_gdf = gpd.sjoin_nearest(self.house_price, self.station_coords)
        if not cols_to_keep:
            cols_to_keep = [
                "Station name",
                "Lat",
                "Lon",
                "geometry",
                "Region_Name",
                "Average_Price",
                "Return intensity",
                "Lower",
                "Upper",
            ]
        self.unified_gdf = self.unified_gdf.loc[:, cols_to_keep].reset_index(drop=True)

    def remove_nans(self):
        """Removes rows from `self.unified_gdf` which the column
        'Average_Price' is NA and resets the index of `self.unified_gdf`.
        NB calling this method will change `self.unified_gdf` in place.

            Returns:
                None
        """
        boolean_mask = self.unified_gdf.Average_Price.isna()
        self.unified_gdf = self.unified_gdf[~boolean_mask].reset_index(drop=True)

    def minmax_normalizer(
        self,
        df,
        risk_var="Return intensity",
        hdi_cols=["Lower", "Upper"],
        upper_bound=0.5,
    ):
        """Min-Max normalizer which normalizes values to have a max
        value of 'upper_bound'. Returns pandas DataFrame of the original
        values and the normalized values.

        Args:
            df (pandas.DataFrame): A pandas.DataFrame containing the data to be normalized.

        Kwargs:
            risk_var (str): The name of the column to be normalized,
                default 'Return intensity'.
            hdi_cols (list): A list of length 2, where each element is a
                string of the name of the columns containing the lower
                and upper HDI, respectively. Default value is
                `['Lower', 'Upper'].
            upper_bound (float): Specify the largest value, effectively
                control the scale of the normalizer.

        Returns:
            adj_multiplier (pandas.DataFrame): A pandas.DataFrame
                containing normalized values.
        """

        range_data = np.ptp(df[risk_var])
        adj_multiplier = df[[risk_var]].apply(
            lambda x: ((x - df[risk_var].min()) / (range_data * (1 / upper_bound)))
        )
        adj_multiplier = adj_multiplier.rename({risk_var: "Adj multiplier"}, axis=1)
        return adj_multiplier

    def get_risk_adjusted_hp(self, df, adjustment_df, threshold=160):
        """This method gets risk-adjusted house prices for stations with
        rainfall above threshold. adjustment_df is a pandas.DataFrame of
        normalized rainfall.

        Args:
            df (pandas.DataFrame): A pandas.DataFrame containing return
                intensity and house prices.
            adjustment_df (pandas.DataFrame): A pandas.DataFrame
                generated by `self.minmax_normalizer`.

        Kwargs:
            threshold (int): The value that 'Return intensity' should
                exceed for risk-adjustment to take place, default value
                160 (mm).

        Returns:
            df (pandas.DataFrame): A pandas.DataFrame which is `df`
                with additional columns containing risk-adjustment in
                addition to a column contain the values used to peform
                risk-adjustment.
        """

        # calculate the risk-adjusted value to add to house price for all rows
        df["adjustment_values"] = df.loc[:, ["Average_Price"]].mul(adjustment_df.values)
        # add risk correction to original house price
        df["Risk-adjusted house price"] = df["adjustment_values"].add(
            df["Average_Price"]
        )
        # for values below threshold, set them to their unadjusted value
        df[["Risk-adjusted house price"]] = df[["Risk-adjusted house price"]].mask(
            lambda x: df["Return intensity"] < threshold, df["Average_Price"], axis=0
        )
        # for values below threshold, set them to zero
        # this column is needed for stacked bar plots
        df[["adjustment_values"]] = df[["adjustment_values"]].mask(
            lambda x: df["Return intensity"] < threshold, 0.0, axis=0
        )
        return df

    def get_error_bounds(
        self,
        df,
        risk_var="Return intensity",
        targ_col="Risk-adjusted house price",
        hdi_cols=["Lower", "Upper"],
    ):
        """This methods takes a df containing the 95% HDI and uses the
        ratio of the lower and upper bounds to calculate error for
        risk-adjusted house prices. It assigns the same ratio between
        mean rainfall prediction and the 95% HDI interval to the
        risk-adjusted house prices

        Args:
            df (pandas.DataFrame): A pandas.DataFrame containing the 95%
                HDI and uses the ratio of the lower and upper bounds to
                calculate error for risk-adjusted house prices.

        Kwargs:
            risk_var (str): The name of the column containing the risk
                variable in `df`.
            targ_col (str): The name of the column continaing the
                risk-adjusted values in `df`.
            hdi_cols (list): A list of length 2 containing the names in
                the HDI columns in `df` order must be lower bound
                followed by upper bound.
        """

        hdi_ratios = df[hdi_cols].div(df[[risk_var]].values)
        hdi_values = hdi_ratios.mul(df.loc[:, [targ_col]].values)
        hdi_values = hdi_values.rename(
            {hdi_cols[0]: hdi_cols[0] + "_adj", hdi_cols[1]: hdi_cols[1] + "_adj"},
            axis=1,
        )
        df = df.join(hdi_values)

        return df

    def get_risk_adj_df(
        self,
        df=None,
        risk_var="Return intensity",
        adj_col="Average_Price",
        hdi_cols=["Lower", "Upper"],
        risk_weight=0.5,
        risk_threshold=160,
    ):
        """Returns risk-adjusted house prices, lower and upper estimates.
        Lower and upper estimates are based on the ratio of the HDI
        lower/upper estimates for risk var. Risk weight determines the
        largest value of the risk factor, risk threshold set the level
        which must be exceeded for risk-adjustment to take place (in
        [mm] rainfall).

        Kwargs:
            df (pandas.DataFrame): A pandas.DataFrame containing the
                risk varible predictions, HDI bounds and house prices.
            risk_var (str): The name of the risk variable in `df`,
                default value 'Return intensity'.
            adj_col (str): The name of the column containing values to
                be risk-adjusted in `df`, default 'Average_Price'.
            risk_weight (float): The weighting that risk_var should
                carry, larger values give a greater weighting. This
                value is passed to self.minmax_normalizer(). Default is
                0.5.
            risk_threshold (int): The value that should be exceeded for
                risk-adjustment to occur, default 160 (mm rainfall).

        Returns:
            df (pandas.DataFrame): A pandas.DataFrame containing
                risk-adjusted values and HDI bounds appendeded to `df`.
        """
        if df is None:
            df = self.unified_gdf.groupby("Station name").first()
        adj_multiplier = self.minmax_normalizer(df, upper_bound=risk_weight)
        df = self.get_risk_adjusted_hp(df, adj_multiplier, risk_threshold)
        df = self.get_error_bounds(df)
        return df

    def tidy_data(self, gev_dict):
        """Performs data munging on a gev_dict instance.

        Returns:
            None
        """
        self.get_metrics_df(gev_dict)
        self.add_metrics_to_stations()
        self.convert_crs()
        self.join_house_prices()
        self.remove_nans()  # keep nans for plotting NaNs on choropleth map
