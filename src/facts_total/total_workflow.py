from typing import List
import xarray as xr
import numpy as np
from pathlib import Path
import click
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
        pyear_start : int
            Start year for projections.
        pyear_end : int
            End year for projections.
        pyear_step : int
            Year step for projections.
        """
        self.name = name
        self.paths_list = paths_list
        self.pyear_start = pyear_start
        self.pyear_end = pyear_end
        self.pyear_step = pyear_step

    def print_files(self):
        """
        Prints the list of file paths in paths_list.
        """
        click.echo("Files to be totaled:")
        for path in self.paths_list:
            click.echo(f"- {path}")

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
            Preprocess function to add a 'file' dimension. This function is applied to each dataset as its read in. It checks that the min/max/step of the years dimension matches the provided pyear values, and adds a 'file' dimension with source info.
            It also adds the filename of the source file as the entry for the file dimension.

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

            # add source info along file dim
            file = ds.encoding["source"]
            parent_dir = Path(file).parent.stem
            fname = Path(file).with_suffix("").name
            source = str(Path(parent_dir) / Path(fname))
            ds["file"] = [source]

            # Check that year steps are uniform over time dim
            ds = ds.expand_dims(["year_step"])
            step = ds["years"].diff("years")

            assert len(np.unique(step.data)) == 1, (
                f"Year steps are not uniform across time dimension. The step values are: {np.unique(step.data)}"
            )
            ds["year_step"] = [np.unique(step.data)[0]]

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
        setattr(self, "combined_ds", combined_ds)

    def format_projections(self) -> xr.Dataset:
        combined_ds = getattr(self, "combined_ds")
        # Check dimensions of each dataset
        if len(np.unique(combined_ds["year_step"])) > 1:
            step_message = click.wrap_text(
                f"⚠️ ⚠️ Year steps are not the same across all datasets. Check default values of --pyear-step in these modules. Received: {np.unique(combined_ds['year_step'].values)}. ⚠️ ⚠️",
                width=70,
            )
            click.echo(step_message)
        # Drop year_step dim after check
        combined_ds = combined_ds.squeeze(dim="year_step", drop=True)
        # Format lat/lon variables (want them to exist along locations dim only, not files)
        # first downcast
        if "locations" not in combined_ds.coords:
            combined_ds = combined_ds.set_coords("locations")
        combined_ds = combined_ds.set_coords(["lat", "lon"])
        coords_ls = ["lat", "lon"]
        for coord in coords_ls:
            combined_ds[coord] = combined_ds[coord].astype("float32")
            combined_ds[coord].load()
        # Ensure that lat/lon do not vary along file dim before dropping
        locations = combined_ds["locations"].values
        if np.isscalar(locations) or locations.ndim == 0:
            locations = [locations.item() if hasattr(locations, "item") else locations]
        else:
            locations = locations.tolist()
        for loc in locations:
            for coord in coords_ls:
                assert (
                    len(np.unique(combined_ds[coord].sel(locations=loc).values)) == 1
                ), (
                    f"{coord} variable varies along 'file' dimension for location {loc}: {np.unique(combined_ds[coord].sel(locations=loc).values)}."
                )

        # detach lat/lon from file dim
        lat_keep = combined_ds.lat.isel(file=0)
        lon_keep = combined_ds.lon.isel(file=0)
        combined_ds = combined_ds.assign_coords(
            lat=("locations", lat_keep.values), lon=("locations", lon_keep.values)
        )
        combined_ds = combined_ds.reset_coords(["lat", "lon"])

        # Format filename data to track cubes included in total
        # this is a hacky (temp) replacement for how its handled in facts1 using
        # os.listdir() for nc files in the experiment output dir
        source_cubes = combined_ds["file"].values.tolist()
        i = 1
        for cube in source_cubes:
            combined_ds.attrs.update({f"cube {i}": cube})
            i += 1
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
        ds_attrs = ds.attrs.copy()

        ds = ds.sum(dim="file")

        # Define the missing value for the netCDF files
        nc_missing_value = np.nan  # np.iinfo(np.int16).min
        ds["sea_level_change"].attrs = {
            "units": "mm",
            "missing_value": nc_missing_value,
        }
        setattr(self, "totaled_ds", ds)
        ds.attrs = ds_attrs
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
            " No totaled dataset found. Please run get_projections first."
        )
        totaled_ds = getattr(self, "totaled_ds")

        # make sure attrs can be written
        encoding = {"sea_level_change": {"zlib": True, "complevel": 4, "dtype": "f4"}}

        totaled_ds.to_netcdf(outpath, encoding=encoding)
        logger.info("Totaled projections written to %s", outpath)
