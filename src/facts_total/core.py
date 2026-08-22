from typing import List
import xarray as xr
import numpy as np
import logging

logger = logging.getLogger(__name__)


class WorkflowTotalerError(Exception):
    pass


CANONICAL_COORD_ORDER = ("samples", "years", "locations")


def normalize_slc_dim_order(ds: xr.Dataset) -> xr.Dataset:
    """
    Transposes the 'sea_level_change' variable of a dataset to match
    CANONICAL_COORD_ORDER, if its dimensions match that order's dimensions.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing a 'sea_level_change' variable.

    Returns
    -------
    xr.Dataset
        Dataset with 'sea_level_change' transposed to the canonical dimension
        order, or unchanged if its dimensions don't match.
    """

    ds["sea_level_change"] = ds["sea_level_change"].transpose(
        ..., *CANONICAL_COORD_ORDER, missing_dims="ignore"
    )
    return ds


def get_projections(
    paths_list: List[str], pyear_start: int, pyear_end: int, pyear_step: int
) -> xr.Dataset:
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

        if (
            ds["years"].min().item() != pyear_start
            or ds["years"].max().item() != pyear_end
        ):
            logger.warning(
                "\n ⚠️ ⚠️ Warning ⚠️ ⚠️: \nThe dataset being processed has a years dimension "
                "from %s to %s, which does not match the provided pyear-start (%s) and "
                "pyear-end (%s). Subsetting dataset to provided pyear values.",
                ds["years"].min().item(),
                ds["years"].max().item(),
                pyear_start,
                pyear_end,
            )

            ds = ds.sel(years=slice(pyear_start, pyear_end))

        step = ds["years"].diff("years")
        if len(np.unique(step.data)) != 1 or np.unique(step.data)[0] != pyear_step:
            logger.warning(
                "\n ⚠️ ⚠️ Warning ⚠️ ⚠️: \nThe dataset being processed has a years dimension with step values %s, which does not match the provided pyear-step (%s). Check that you did not make a mistake specifying the totaling command or the individual modules.",
                np.unique(step.data),
                pyear_step,
            )

        ds = ds.expand_dims("file")

        # add source info along file dim
        file = ds.encoding["source"]
        ds["file"] = [file]

        # Check that year steps are uniform over time dim
        ds = ds.expand_dims(["year_step"])
        step = ds["years"].diff("years")

        assert len(np.unique(step.data)) == 1, (
            f"Year steps are not uniform across time dimension. The step values are: {np.unique(step.data)}"
        )
        ds["year_step"] = [np.unique(step.data)[0]]

        # reorder dims

        ds = normalize_slc_dim_order(ds)
        core = tuple(
            d for d in ds["sea_level_change"].dims if d in CANONICAL_COORD_ORDER
        )
        assert core == CANONICAL_COORD_ORDER, (
            f"Received {core}, expected {CANONICAL_COORD_ORDER}"
        )

        return ds

    combined_ds = xr.open_mfdataset(
        paths_list,
        concat_dim="file",
        combine="nested",
        join="outer",  # may want to change to join='exact'
        preprocess=preprocess_fn,
        chunks="auto",
    )
    return combined_ds


def format_projections(combined_ds: xr.Dataset) -> xr.Dataset:
    # check that year steps are uniform
    if len(np.unique(combined_ds["year_step"])) > 1:
        raise WorkflowTotalerError(
            "Year steps are not the same across all datasets. Check default values "
            "of --pyear-step in these modules. Received: {}".format(
                np.unique(combined_ds["year_step"].values)
            )
        )
    # drop year step dim after check
    combined_ds = combined_ds.squeeze(dim="year_step", drop=True)
    # format la/lon variables (want them to exist along locations dim only, not files)
    if "locations" not in combined_ds.coords:
        combined_ds = combined_ds.set_coords("locations")
    combined_ds = combined_ds.set_coords(["lat", "lon"])
    coord_ls = ["lat", "lon"]
    for coord in coord_ls:
        combined_ds[coord] = combined_ds[coord].astype("float32")
        combined_ds[coord].load()

    # detach lat lon from file dim
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

    # Add sources files to attrs for tracking
    combined_ds.attrs.update(
        {
            "source": "FACTS2: Post-processing total among available contributors: {}".format(
                ",".join(source_cubes)
            )
        }
    )
    # Normalize dimension order
    combined_ds = normalize_slc_dim_order(combined_ds)

    # check that order is correct
    dim_order = tuple(
        d for d in combined_ds["sea_level_change"].dims if d in CANONICAL_COORD_ORDER
    )
    assert dim_order == CANONICAL_COORD_ORDER, (
        f"Received ds with dims in following order: {dim_order}, expected {CANONICAL_COORD_ORDER}"
    )
    return combined_ds


def total_projections(ds: xr.Dataset, workflow_name: str) -> xr.Dataset:
    # make copy of attrs
    attrs = ds.attrs.copy()

    ds = ds.sum(dim="file")

    # Define missing value for netCDFs
    nc_missing_value = np.nan
    ds["sea_level_change"].attrs = {
        "units": "mm",
        "missing_value": nc_missing_value,
        "workflow_name": workflow_name,
    }
    ds.attrs = attrs
    return ds
