from typing import List
import xarray as xr
import numpy as np
import click


class WorkflowTotaler:
    """
    Handles totaling of sealevel projections from modules included in a workflow.

    Attributes
    ----------
    name : str
        Name of the workflow.
    paths_list : list of str
        List of file paths to component-level projection datasets.
    projections_ds : xr.Dataset, optional
        Combined projections dataset, set after calling get_projections().
    totaled_ds : xr.Dataset, optional
        Totaled projections dataset, set after calling total_projections().
    """

    def __init__(
        self,
        name: str,
        paths_list: List[str],
        pyear_start: int,
        pyear_end: int,
        pyear_step: int,
    ):
        """
        Initialize WorkflowTotaler.

        Parameters
        ----------
        name : str
            Name of the workflow.
        paths_list : list of str
            List of file paths to component-level projection datasets.
        """
        self.name = name
        self.paths_list = paths_list
        self.pyear_start = pyear_start
        self.pyear_end = pyear_end
        self.pyear_step = pyear_step

    def get_projections(self) -> xr.Dataset:
        """
        Reads in component-level projection datasets from NetCDF files and combines them
        along a 'file' dimension that is added to each dataset.

        Returns
        -------
        xr.Dataset
            Combined projections dataset with a new 'file' dimension.

        Raises
        ------
        AssertionError
            If 'paths_list' attribute is missing.
        """

        def preprocess_fn(ds: xr.Dataset) -> xr.Dataset:
            """
            Minimal preprocess function to add a 'file' dimension.

            Parameters
            ----------
            ds : xr.Dataset
                Input dataset.

            Returns
            -------
            xr.Dataset
                Dataset with added 'file' dimension and transposed dimensions.
            """

            # check this files dims against the provided pyear values
            pyear_start = self.pyear_start
            pyear_end = self.pyear_end
            pyear_step = self.pyear_step

            if (
                ds["years"].min().item() != pyear_start
                or ds["years"].max().item() != pyear_end
            ):
                message = click.wrap_text(
                    f"⚠️ ⚠️ Warning ⚠️ ⚠️: The dataset being processed has a years dimension from {ds['years'].min().item()} to {ds['years'].max().item()}, which does not match the provided pyear-start ({pyear_start}) and pyear-end ({pyear_end}). Subsetting dataset to provided pyear values.",
                    width=70,
                )
                ds = ds.sel(years=slice(pyear_start, pyear_end))

            step = ds["years"].diff("years")
            if len(np.unique(step.data)) != 1 or np.unique(step.data)[0] != pyear_step:
                message = click.wrap_text(
                    f"⚠️ ⚠️ Warning ⚠️ ⚠️: The dataset being processed has a years dimension with step values {np.unique(step.data)}, which does not match the provided pyear-step ({pyear_step}). Check that you did not make a mistake specifying the totaling command or the individual modules.",
                    width=70,
                )
                click.echo(message)

            ds = ds.expand_dims("file")
            ds["file"] = ["abc"]
            ds = ds.expand_dims(["start_year", "end_year", "year_step"])
            ds["start_year"] = [ds["years"].min().item()]
            ds["end_year"] = [ds["years"].max().item()]
            step = ds["years"].diff("years")
            # Make sure year steps are uniform across time dim
            assert len(np.unique(step.data)) == 1, (
                f"Year steps are not uniform across time dimension. The step values are: {np.unique(step.data)}"
            )
            ds["year_step"] = [np.unique(step.data)[0]]

            # dims_ls = ['years','locations','file','samples']
            # ds = ds.transpose(*dims_ls)
            return ds

        assert hasattr(self, "paths_list"), (
            "WorkflowTotaler object must have 'paths_list' attribute."
        )

        combined_ds = xr.open_mfdataset(
            self.paths_list,
            concat_dim="file",
            combine="nested",
            join="outer",  # may want to change to join='exact'
            preprocess=preprocess_fn,
            chunks="auto",
        )
        # Check dimensions of each dataset
        if len(np.unique(combined_ds["start_year"])) > 1:
            start_message = click.wrap_text(
                f"⚠️ ⚠️ Start years are not the same across all datasets. Check default values of --pyear-start in these modules. Received: {np.unique(combined_ds['start_year'].values)}. ⚠️ ⚠️",
                width=70,
            )
            click.echo(start_message)
        if len(np.unique(combined_ds["end_year"])) > 1:
            end_message = click.wrap_text(
                f"⚠️ ⚠️ End years are not the same across all datasets. Check default values of --pyear-end in these modules. Received: {np.unique(combined_ds['end_year'].values)}. ⚠️ ⚠️",
                width=70,
            )
            click.echo(end_message)
        if len(np.unique(combined_ds["year_step"])) > 1:
            step_message = click.wrap_text(
                f"⚠️ ⚠️ Year steps are not the same across all datasets. Check default values of --pyear-step in these modules. Received: {np.unique(combined_ds['year_step'].values)}. ⚠️ ⚠️",
                width=70,
            )
            click.echo(step_message)

        setattr(self, "projections_ds", combined_ds)
        return combined_ds

    def total_projections(self) -> xr.Dataset:
        """
        Totals projections along the 'file' dimension added in get_projections().

        Returns
        -------
        xr.Dataset
            Dataset with an added 'totaled_sea_level_change' variable.

        Raises
        ------
        AssertionError
            If projections dataset has not been read in.
        """
        # Make sure projections have been read in
        assert hasattr(self, "projections_ds"), (
            "No projections dataset found. Please run get_projections first."
        )
        ds = getattr(self, "projections_ds")

        ds["totaled_sea_level_change"] = ds["sea_level_change"].sum(dim="file")

        ds_keep = ds[['totaled_sea_level_change','lon','lat']]

        setattr(self, "totaled_ds", ds_keep)
        return ds

    def write_totaled_projections(self, outpath: str):
        """
        Writes the totaled projections to a NetCDF file.

        Parameters
        ----------
        outpath : str
            Path to write the NetCDF file to.

        Raises
        ------
        AssertionError
            If totaled dataset has not been created.
        """
        assert hasattr(self, "totaled_ds"), (
            "No totaled dataset found. Please run get_projections first."
        )
        totaled_ds = getattr(self, "totaled_ds")
        totaled_ds.to_netcdf(outpath)
