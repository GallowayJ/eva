import arviz as az
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, gb_outline_gdf, gev_dict, data_dict):
        self.gb_outline_gdf = gb_outline_gdf.dissolve()
        self.gev_dict = gev_dict
        self.data_dict = data_dict

    def plot_ts(self, station_name, fig, axes):
        """Plots the time series with annual maxima highlighted and the
        probability density function estimated using Gaussian kernals ."""

        fig, axes = fig, axes
        df = self.data_dict[station_name]
        df.plot(kind="density", ax=axes[1])
        df.groupby(df.index.year).max().plot(kind="density", ax=axes[1])
        max_idx = df.groupby(df.index.year).idxmax()
        max_val = df.groupby(df.index.year).max()
        axes[0].plot(df.index, df.values, label="Rainfall")
        axes[0].plot(max_idx, max_val, marker="x", ls="none", label="Annual maxima")
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
        """Plot samples from the priors"""

        az.plot_ppc(idata, group="prior", ax=ax)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return ax

    def plot_posterior(self, idata, ax):
        """Plots posterior check."""

        az.plot_ppc(idata, num_pp_samples=100, ax=ax)
        return ax

    def plot_return_period(self, post_pred, return_periods, ax):
        """docstring"""

        y_data = az.summary(
            post_pred.posterior_predictive, var_names="ri", hdi_prob=0.95
        )
        x = return_periods
        y = y_data["mean"]
        y_lower_conf = y_data["hdi_2.5%"]
        y_upper_conf = y_data["hdi_97.5%"]
        ax.plot(x, y, label="Mean prediction")
        ax.fill_between(x, y_lower_conf, y_upper_conf, alpha=0.15, label="95% HDI")
        ax.legend()
        return ax

    def plot_prior_post_prediction(self, station_name, fig, axes):
        """Plots the prior predictive, the posterior predictive and predictions"""

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
        """docstring"""
        gdf = geomunger.unified_gdf
        risk_gdf = geomunger.get_risk_adj_df()
        plotting_kwargs = {
            "scheme": "quantiles",
            "legend": True,
            "cmap": "OrRd",
            "legend_kwds": {"fmt": "{:,.0f}"},
            "lw": 0.3,
        }
        # geopandas .plot() returns a new Axes instance, use this to modify legend etc.
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

        ax_1 = gdf.plot(**plotting_kwargs, column="Average_Price", ax=axes[1], ec="k")
        new_labels = [
            label.get_text().replace(", ", " - ", 1)
            for label in ax_1.get_legend().get_texts()
        ]
        for idx, label in enumerate(new_labels):
            ax_1.get_legend().get_texts()[idx].set_text(label)

        df_risk_adj_hp = gdf.join(risk_gdf["Risk-adjusted house price"], on="Station name")
        ax_2 = df_risk_adj_hp.plot(
            **plotting_kwargs, column="Risk-adjusted house price", ax=axes[2], ec="k"
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
            self.gb_outline_gdf.plot(fc="none", ec="k", ax=axes[0], lw=0.2, zorder=0)

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
        """Helper method to get the absolute difference between values and hdi bounds
        as the absolute difference and not the values themselves are needed for
        matplotlib.pyplot.axes.errrorbar."""

        error_bar_df = df.loc[:, hdi_cols].sub(df.loc[:, [targ_col]].values).abs()
        return error_bar_df

    def plot_risk_adj_bar(self, df, fig, ax, index=None):
        """Plot risk adjusted house prices and error bars, df is the
        result of GeoMunger.get_risk_adj_df()."""
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
