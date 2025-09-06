from io import StringIO
import os.path
import time

from bs4 import BeautifulSoup
import geopandas as gpd
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import re
import requests


class WebScraper:
    def __init__(self, base_url):
        self.base_url = base_url
        self.station_links = {}
        self.station_dict = {}
        # List of stations in an unparseable format.
        self.nonstandard_list = []

    def get_stations_and_links(self):
        page = requests.get(self.base_url)
        soup = BeautifulSoup(page.content, features="html.parser")
        # skip first element (not a station)
        self.station_links = soup.select("option")[1:]

        # Create dictionary where keys are station names and values are a list
        # of length 1 containing the link to station historical data.
        for each_tag in self.station_links:
            station_name = each_tag.text
            station_link = each_tag.attrs["value"]
            self.station_dict.update({station_name: [station_link]})
        return

    def get_coordinates(self):
        """Attempt to extract coordinates from station metadata. If this fails,
        add the station name to the nonstandard_list instance attribute."""
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
        return

    def get_data(self):
        self.get_stations_and_links()
        # get station data, parse into pandas DataFrames and retrieve station metadata
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
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.station_coords_list = []

    def drop_unneeded_cols(self, target_col=["Rainfall"]):
        for k, v in self.data_dict.items():
            df = v[1]
            cols_to_keep = ["Year", "Month"]
            cols_to_keep.extend(target_col)
            self.data_dict[k][1] = df[cols_to_keep]
        return self.data_dict

    def remove_units_header(self, target_str="mm", target_col="Rainfall"):
        """If the first row of values has been used for units of measure, remove
        this row."""
        for k, v in self.data_dict.items():
            df = v[1]
            if df.loc[0, target_col] == target_str:
                self.data_dict[k][1] = df.iloc[1:].reset_index(drop=True)
                self.data_dict[k][1] = df.dropna(how="all")
        return

    def strip_asterices(self):
        for k, v in self.data_dict.items():
            df = v[1]
            self.data_dict[k][1] = df.replace(r"\*", "", regex=True)
        return

    def get_station_coords(self):
        for k, v in self.data_dict.items():
            lat, lon = v[3], v[4]
            self.station_coords_list.append([k, lat, lon])
        return

    def assign_correct_dtypes(self):
        for k, v in self.data_dict.items():
            self.data_dict[k][1] = self.data_dict[k][1].dropna(how="any")
            self.data_dict[k][1] = self.data_dict[k][1].apply(
                pd.to_numeric, axis=1, errors="coerce"
            )
            self.data_dict[k][1] = self.data_dict[k][1].astype(
                {"Year": "Int64", "Month": "Int64", "Rainfall": "float32"}
            )
        return

    def create_dt_index(self):
        for k, v in self.data_dict.items():
            v[1]["Date"] = pd.to_datetime(
                dict(year=v[1]["Year"], month=v[1]["Month"], day=1)
            ) + MonthEnd(0)
            self.data_dict[k][1] = v[1].set_index("Date", drop=True)
            cols_to_drop = ["Year", "Month"]
            self.data_dict[k][1] = self.data_dict[k][1].drop(cols_to_drop, axis=1)
        return

    def get_extreme_dict(self, data_dict):
        """This method takes a dictionary of pandas DataFrames and resamples
        by year, returning the maxima. It returns a dictionary containing
        the resampled DataFrames."""
        maxima_dict = data_dict.copy()
        for each_site, each_df in maxima_dict.items():
            maxima_dict[each_site] = each_df.groupby(each_df.index.year).max()
            idx_max_dates = each_df.groupby(each_df.index.year).idxmax()
            # this column contain datetime of when maximum occured in each period
            maxima_dict[each_site] = maxima_dict[each_site].set_index(
                idx_max_dates["Rainfall"], drop=True
            )
            return maxima_dict

    def tidy_data(self):
        self.drop_unneeded_cols()
        self.remove_units_header()
        self.strip_asterices()
        self.assign_correct_dtypes()
        self.create_dt_index()
        self.get_station_coords()
        return

    def save_data(self, directory, **kwargs):
        """This method saves this objects data_dict to .csv files using
        pandas.to_csv() methods. **kwargs are passed to pandas.to_csv()."""
        for k, v in self.data_dict.items():
            fp = os.path.join(directory, f"{k}.csv")
            v[1].to_csv(fp, **kwargs)
        return


class GeoMunger:
    def __init__(self, gdf_house_price, gdf_station_coords, aoi_outline=None):
        self.house_price = gdf_house_price
        self.station_coords = gdf_station_coords
        self.metrics_df = None
        self.unified_gdf = None
        self.aoi_outline = aoi_outline
        # self.risk_factor = None

    def extract_station_metrics(self, gev_dict):
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
        """This method takes a pandas DataFrame containing the predicted return
        period, intensity and confidence intervals and returns a single row DataFrame
        containing this metric in addtion to adding a column with the site name."""

        df = df.iloc[[(df["Return period"] - return_period_yrs).abs().argsort()[0]]]
        df.insert(0, "Station name", station_name)
        return df

    def get_metrics_df(self, gev_dict):
        """This methods takes a dict of the GEV fit results"""

        metrics_dict = {k: self.extract_station_metrics(v) for k, v in gev_dict.items()}
        metrics_dict = {
            k: self.get_return_period_results(k, v) for k, v in metrics_dict.items()
        }
        self.metrics_df = pd.concat(metrics_dict, ignore_index=True)

    def add_metrics_to_stations(self):
        self.station_coords = self.station_coords.merge(
            self.metrics_df, on="Station name"
        )

    def convert_crs(self, crs="EPSG:27700"):
        self.house_price = self.house_price.to_crs(crs)
        self.station_coords = self.station_coords.to_crs(crs)

    def join_house_prices(self, cols_to_keep=None):
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
        boolean_mask = self.unified_gdf.Average_Price.isna()
        self.unified_gdf = self.unified_gdf[~boolean_mask].reset_index(drop=True)

    def minmax_normalizer(
        self,
        df,
        risk_var="Return intensity",
        hdi_cols=["Lower", "Upper"],
        upper_bound=0.5,
    ):
        """Min-Max normalizer which normalizes values to have a max value of
        'upper_bound'. Returns pandas DataFrame of the original values and the
        normalized values."""

        range_data = np.ptp(df[risk_var])
        adj_multiplier = df[[risk_var]].apply(
            lambda x: ((x - df[risk_var].min()) / (range_data * (1 / upper_bound)))
        )
        adj_multiplier = adj_multiplier.rename({risk_var: "Adj multiplier"}, axis=1)
        return adj_multiplier

    def get_risk_adjusted_hp(self, df, adjustment_df, threshold=160):
        """This method gets risk-adjusted house prices for stations with rainfall
        above threshold. adjustment_df is a pandas.DataFrame of normalized rainfall."""

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
        """This methods takes a df containing the 95% HDI and uses the ratio of the
        lower and upper bounds to calculate error for risk-adjusted house prices.
        It assigns the same ratio between mean rainfall prediction and the 95% HDI
        interval to the risk-adjusted house prices"""

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
        """Returns risk-adjusted house prices, lower and upper estimates. Lower and
        upper estimates are based on the ratio of the HDI lower/upper estimates for
        risk var. Risk weight determines the largest value of the risk factor, riskt
        threshold set the level which must be exceeded for risk-adjustment to take
        place (in [mm] rainfall)."""
        if df is None:
            df = self.unified_gdf.groupby("Station name").first()
        adj_multiplier = self.minmax_normalizer(df, upper_bound=risk_weight)
        df = self.get_risk_adjusted_hp(df, adj_multiplier, risk_threshold)
        df = self.get_error_bounds(df)
        return df

    def tidy_data(self, gev_dict):
        self.get_metrics_df(gev_dict)
        self.add_metrics_to_stations()
        self.convert_crs()
        self.join_house_prices()
        self.remove_nans()  # keep nans for plotting NaNs on choropleth map
