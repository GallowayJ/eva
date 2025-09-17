import arviz as az
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """Creates visualizations. In general this class takes data and
    instances of matplotlib.figure.Figure and matplotlib.axes.Axes, it
    will create a visualization and return the modified fig and ax(es)
    instances for further fine tuning.

    Attributes:
        self.gb_outline_gdf (geopandas.GeoDataFrame): An outline of the
            island of Great Britain.
        self.gev_dict (dict): The results of statiscal modelling created
            by the accompanying Jupyter notebook.
        self.data_dict (dict): The instance of `data_dict` created by
            DataImportUtils, containing data to be used in visualizations.
    """

    def __init__(self, gb_outline_gdf, gev_dict, data_dict):
        """Construct an instance of Plotter.
        Args:
            self.gb_outline_gdf (geopandas.GeoDataFrame): An outline of
                the island of Great Britain.
            self.gev_dict (dict): The results of statiscal modelling
                created by the accompanying Jupyter notebook.
            self.data_dict (dict): The instance of `data_dict` created
                by DataImportUtils, containing data to be used in visualizations.
        Returns:
            None
        """
        self.gb_outline_gdf = gb_outline_gdf.dissolve()
        self.gev_dict = gev_dict
        self.data_dict = data_dict

    def plot_ts(self, station_name, fig, axes):
        """Plots the time series with annual maxima highlighted and the
        probability density function estimated using Gaussian kernals.

        Args:
            station_name (str): A string of the met station name.
            fig (matplotlib.figure.Figure): An instance of
                matplotlib.pyplot.Fig.
            axes (matplotlib.axes.Axes or array of Axes): An instance of
                matplotlib.axes.Axes.

        Returns:
            fig (matplotlib.figure.Figure): The modified instance of
                matplotlib.pyplot.Fig.
            axes (matplotlib.axes.Axes or array of Axes): The modified
                array of Axes of matplotlib.axes.Axes instances.
        """
        fig, axes = fig, axes
        df = self.data_dict[station_name]
        df.plot(kind="density", ax=axes[1])
        df.groupby(df.index.year).max().plot(kind="density", ax=axes[1])
        max_idx = df.groupby(df.index.year).idxmax()
        max_val = df.groupby(df.index.year).max()
        axes[0].plot(df.index, df.values, label="Rainfall")
        axes[0].plot(
            max_idx, max_val, marker="x", ls="none", label="Annual maxima"
        )
        axes[0].set_title("Time series")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Rainfall [mm]")
        axes[0].legend()
        axes[1].set_title("PDF estimate (using Gaussian kernals)")
        axes[1].set_xlabel("Rainfall [mm]")
        axes[1].legend(["Rainfall", "Extreme events"])
        fig.suptitle(f"{station_name}")
        return fig, axes

    def plot_prior_check(self, idata, ax, xlim=[0, 180], ylim=[0, 0.4]):
        """Plots samples from the priors

        Args:
            idata (arviz.InferenceData): An instance of
                arviz.InferenceData.
            ax (matplotlib.axes.Axes): An instance of
                matplotlib.axes.Axes.

        Kwargs:
            xlim (list): a list to be passed to matplotlib.pyplot.xlim,
                default value `[0, 180]`.
            ylim (list): a list to be passed to matplotlib.pyplot.ylim,
                default value `[0, 0.4]`.

        Returns:
            ax (matplotlib.axes.Axes): The modified instance of
                matplotlib.axes.Axes.
        """
        az.plot_ppc(idata, group="prior", ax=ax)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return ax

    def plot_posterior(self, idata, ax):
        """Plots posterior predictive check.

        Args:
            idata (arviz.InferenceData): An instance of
                arviz.InferenceData.
            ax (matplotlib.axes.Axes): An instance of
                matplotlib.axes.Axes.

        Returns:
            ax (matplotlib.axes.Axes): The modified instance of
                matplotlib.axes.Axes.
        """
        az.plot_ppc(idata, num_pp_samples=100, ax=ax)
        return ax

    def plot_return_period(self, post_pred, return_periods, ax):
        """Plots a graph of the return intensity for each return period
        with 95% HDI.

        Args:
            post_pred (arviz.InferenceData.posterior): An instance of
                arviz.InferenceData.posterior.
            return_periods (numpy.array): The numpy array used to
                generate predictions for the posterior.
            ax (matplotlib.axes.Axes): An instance of matplotlib.axes.Axes.

        Returns:
            ax (matplotlib.axes.Axes): The modified instance of
                matplotlib.axes.Axes.
        """
        y_data = az.summary(
            post_pred.posterior_predictive, var_names="ri", hdi_prob=0.95
        )
        x = return_periods
        y = y_data["mean"]
        y_lower_conf = y_data["hdi_2.5%"]
        y_upper_conf = y_data["hdi_97.5%"]
        ax.plot(x, y, label="Mean prediction")
        ax.fill_between(
            x, y_lower_conf, y_upper_conf, alpha=0.15, label="95% HDI"
        )
        ax.legend()
        return ax

    def plot_prior_post_prediction(self, station_name, fig, axes):
        """Plots the prior predictive, the posterior predictive and
        predictions. Styles plots using climatex.mplstyle (hard-coded).

        Args:
            station_name (str): A string of the station name key in
                gev_dict.
            fig (matplotlib.figure.Figure): An instance of
                matplotlib.pyplot.Fig.
            axes (an array of matplotlib.axes.Axes): The array of
                matplotlib.axes.Axes.

        Returns:
            fig (matplotlib.figure.Figure): The modified instance of
                matplotlib.pyplot.Fig.
            axes (matplotlib.axes.Axes or array of Axes): The modified
                array of Axes of matplotlib.axes.Axes instances.
        """

        plt.style.use("./climatex.mplstyle")
        site_idata = self.gev_dict[station_name]["idata"]
        post_pred = self.gev_dict[station_name]["post_pred"]
        return_periods = self.gev_dict[station_name]["return periods"]
        axes_titles = [
            "Prior predictive check",
            "Posterior predictive check",
            "Posterior predictions",
        ]
        self.plot_prior_check(site_idata, axes[0])
        self.plot_posterior(site_idata, axes[1])
        self.plot_return_period(post_pred, return_periods, axes[2])

        for idx, title in enumerate(axes_titles):
            axes[idx].set_title(title)

        ylimit = axes[1].get_ylim()[1]
        axes[0].set_ylim([None, ylimit])
        for ax in axes[:2]:
            ax.set_xlim([20, 400])
            ax.set_xlabel("Rainfall [mm]")
            ax.set_ylabel("Density")
            ax.set_yticks(np.linspace(0, ylimit, 4))

        axes[2].set_ylabel("Rainfall [mm]")
        axes[2].set_xlabel("Return period [yr]")
        fig.suptitle(f"{station_name} station")
        plt.subplots_adjust(wspace=0.3)
        return fig, axes

    def plot_choropleth(self, geomunger, fig, axes):
        """Plots 3 choropleth maps in one figure, the first of `Return
        intensity`, the second is of the risk variable (average house
        price), and the third is of the risk-adjusted house price.

        Args:
            geomunger (class.GeoMunger): An instance of GeoMunger.
            fig (matplotlib.figure.Figure): An instance of
                matplotlib.pyplot.Fig.
            axes (an array of matplotlib.axes.Axes): The array of
                matplotlib.axes.Axes.

        Returns:
            fig (matplotlib.figure.Figure): The modified instance of
                matplotlib.pyplot.Fig.
            axes (matplotlib.axes.Axes or array of Axes): The modified
                array of Axes of matplotlib.axes.Axes instances.
        """
        gdf = geomunger.unified_gdf
        risk_gdf = geomunger.get_risk_adj_df()
        plotting_kwargs = {
            "scheme": "quantiles",
            "legend": True,
            "cmap": "OrRd",
            "legend_kwds": {"fmt": "{:,.0f}"},
            "lw": 0.3,
        }
        # geopandas .plot() returns a new Axes instance, use this to
        # modify legend etc.
        ax_0 = gdf.plot(
            **plotting_kwargs, column="Return intensity", ax=axes[0], ec="k"
        )
        new_labels = [
            label.get_text().replace(", ", " - ", 1)
            for label in ax_0.get_legend().get_texts()
        ]
        for idx, label in enumerate(new_labels):
            ax_0.get_legend().get_texts()[idx].set_text(label)
        handles, labels = ax_0.get_legend_handles_labels()

        ax_1 = gdf.plot(
            **plotting_kwargs, column="Average_Price", ax=axes[1], ec="k"
        )
        new_labels = [
            label.get_text().replace(", ", " - ", 1)
            for label in ax_1.get_legend().get_texts()
        ]
        for idx, label in enumerate(new_labels):
            ax_1.get_legend().get_texts()[idx].set_text(label)

        df_risk_adj_hp = gdf.join(
            risk_gdf["Risk-adjusted house price"], on="Station name"
        )
        ax_2 = df_risk_adj_hp.plot(
            **plotting_kwargs,
            column="Risk-adjusted house price",
            ax=axes[2],
            ec="k",
        )
        new_labels = [
            label.get_text().replace(", ", " - ", 1)
            for label in ax_2.get_legend().get_texts()
        ]
        for idx, label in enumerate(new_labels):
            ax_2.get_legend().get_texts()[idx].set_text(label)

        # plot station locations
        for each_ax in axes.flatten():
            geomunger.station_coords.plot(
                ax=each_ax, zorder=3, marker="x", color="green"
            )

        # plot GB outline and n/a
        for each_ax in axes.flatten():
            self.gb_outline_gdf.plot(
                hatch="//", ec="r", fc="none", ax=each_ax, zorder=-1
            )
            self.gb_outline_gdf.plot(
                fc="none", ec="k", ax=axes[0], lw=0.2, zorder=0
            )

        axes[0].set_title("Intensity of 20-year Rainfall event [mm]")
        axes[1].set_title("Average house price [£]")
        axes[2].set_title("Risk-adjusted average house price [£]")
        # styling
        for each_ax in axes.flatten():
            each_ax.get_xaxis().set_visible(False)
            each_ax.get_yaxis().set_visible(False)

        # create proxy artists for figure legend
        na_legend_patch = mpatches.Patch(
            fc="none", color="r", hatch="///", label="Missing data"
        )
        met_station = axes[0].plot(
            [], [], marker="x", c="green", ls="none", label="Met station"
        )
        fig.legend(
            handles=[na_legend_patch, met_station[0]],
            loc="outside upper center",
            bbox_to_anchor=(0.5, 0.97),
        )
        return fig, axes

    def get_difference(
        self,
        df,
        targ_col="Risk-adjusted house price",
        hdi_cols=["Lower_adj", "Upper_adj"],
    ):
        """Helper method to get the absolute difference between values
        and hdi bounds as the absolute difference and not the values
        themselves are needed for matplotlib.pyplot.axes.errrorbar.

        Args:
            df (pandas.DataFrame): A pandas.DataFrame of the
                risk-adjusted values generated by DataImportUtils.

        Kwargs:
            targ_col (str): A string of the name of the column
                containing risk-adjusted values, default 'Risk-adjusted
                house price'.
            hdi_cols (list): A list of length 2 containing the name of
                the columns with the risk-adjusted lower and upper
                bounds, respectively. Default ['Lower_adj', 'Upper_adj'].

        Returns:
            error_bar_df (pandas.DataFrame): A pandas.DataFrame
                containing data used to add error bars to plots.
        """
        error_bar_df = (
            df.loc[:, hdi_cols].sub(df.loc[:, [targ_col]].values).abs()
        )
        return error_bar_df

    def plot_risk_adj_bar(self, df, fig, ax, index=None):
        """Plots risk-adjusted house prices and error bars, df is the
        result of GeoMunger.get_risk_adj_df().

        Args:
            df (pandas.DataFrame): A pandas.DataFrame containing
                risk-adjusted house prices and statistical predictions.
            fig (matplotlib.figure.Figure): An instance of
                matplotlib.pyplot.Fig.
            axes (matplotlib.axes.Axes): An instance of
                matplotlib.axes.Axes.

        Kwargs:
            index (pandas.DataFrame.index): An instance of
                pandas.DataFrame.index to be used for the y-axis (NB
                this is a horizontal bar plot, so these are station
                names).

        Returns:
            fig (matplotlib.figure.Figure): The modified instance of
                matplotlib.pyplot.Fig.
            axes (matplotlib.axes.Axes): The modified
                matplotlib.axes.Axes instance.
            df.index (pandas.DataFrame.index): The
                pandas.DataFrame.index used a labels for the plot, these
                are station names.
        """
        if index is not None:
            df.index = df.loc[index].index

        ax.barh(df.index, df["Average_Price"])
        ax.barh(df.index, df["adjustment_values"], left=df["Average_Price"])
        ax.errorbar(
            df["Risk-adjusted house price"],
            df.index,
            color="none",
            ecolor="k",
            xerr=self.get_difference(df).T,
            label="Uncertainty",
        )
        ax.legend(["House price", "Risk-adjustment", "Error"])
        ax.set_title("Risk-adjusted house prices")
        return fig, ax, df.index
