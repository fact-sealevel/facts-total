import logging

import numpy as np
import pytest
import xarray as xr

from facts_total.core import (
    CANONICAL_COORD_ORDER,
    WorkflowTotalerError,
    format_projections,
    get_projections,
    normalize_slc_dim_order,
    total_projections,
)


def make_dataset(dim_order):
    """Helper function to create minimal dataset"""

    sizes = {"samples": 100, "years": 15, "locations": 1}
    shape = [sizes[d] for d in dim_order]
    return xr.Dataset(
        {"sea_level_change": xr.DataArray(np.ones(shape), dims=dim_order)},
        coords={
            "samples": np.arange(100),
            "years": np.arange(2020, 2161, 10),
            "locations": np.array([12]),
        },
    )


def make_projections_dataset(dim_order, n_files=2, year_step_values=(10,)):
    """Helper function to create a minimal dataset resembling the combined
    output of get_projections, suitable as input to format_projections"""

    sizes = {"samples": 100, "years": 15, "locations": 1, "file": n_files}
    shape = [sizes[d] for d in dim_order]
    n_locations = sizes["locations"]
    lat_values = (
        np.arange(n_files * n_locations).reshape(n_files, n_locations).astype(float)
    )
    lon_values = lat_values + 100
    return xr.Dataset(
        {
            "sea_level_change": xr.DataArray(np.ones(shape), dims=dim_order),
            "lat": xr.DataArray(lat_values, dims=("file", "locations")),
            "lon": xr.DataArray(lon_values, dims=("file", "locations")),
        },
        coords={
            "samples": np.arange(sizes["samples"]),
            "years": np.arange(2020, 2161, 10),
            "locations": np.array([12]),
            "file": [f"file{i}.nc" for i in range(n_files)],
            "year_step": np.array(year_step_values),
        },
    )


def write_component_dataset(path, years, dim_order=("years", "samples", "locations")):
    """Helper function to write a minimal component-level projections dataset
    to a netcdf file, suitable as input to get_projections"""

    sizes = {"samples": 100, "years": len(years), "locations": 1}
    shape = [sizes[d] for d in dim_order]
    ds = xr.Dataset(
        {"sea_level_change": xr.DataArray(np.ones(shape), dims=dim_order)},
        coords={
            "samples": np.arange(sizes["samples"]),
            "years": np.array(years),
            "locations": np.array([12]),
        },
    )
    ds.to_netcdf(path)
    return path


def test_normalize_dim_order_nonstandard():
    ds = make_dataset(("years", "samples", "locations"))
    result = normalize_slc_dim_order(ds)
    assert result["sea_level_change"].dims == ("samples", "years", "locations")


def test_normalize_dim_order_already_canonical():
    ds = make_dataset(("samples", "years", "locations"))
    result = normalize_slc_dim_order(ds)
    assert result["sea_level_change"].dims == ("samples", "years", "locations")


def test_normalize_dim_data_unchanged():
    ds = make_dataset(("years", "locations", "samples"))
    result = normalize_slc_dim_order(ds)
    np.testing.assert_array_equal(
        result["sea_level_change"].values,
        ds["sea_level_change"].transpose("samples", "years", "locations").values,
    )


def test_format_projections_returns_ds_with_correct_dim_order():
    ds = make_projections_dataset(("file", "years", "samples", "locations"))
    result = format_projections(ds)
    dims = tuple(
        d for d in result["sea_level_change"].dims if d in CANONICAL_COORD_ORDER
    )
    assert dims == CANONICAL_COORD_ORDER


def test_format_projections_drops_year_step_dim():
    ds = make_projections_dataset(("file", "years", "samples", "locations"))
    result = format_projections(ds)
    assert "year_step" not in result.dims


def test_format_projections_detaches_lat_lon_from_file_dim():
    ds = make_projections_dataset(("file", "years", "samples", "locations"))
    result = format_projections(ds)
    assert result["lat"].dims == ("locations",)
    assert result["lon"].dims == ("locations",)
    np.testing.assert_array_equal(result["lat"].values, ds["lat"].isel(file=0).values)
    np.testing.assert_array_equal(result["lon"].values, ds["lon"].isel(file=0).values)


def test_format_projections_casts_lat_lon_to_float32():
    ds = make_projections_dataset(("file", "years", "samples", "locations"))
    result = format_projections(ds)
    assert result["lat"].dtype == np.float32
    assert result["lon"].dtype == np.float32


def test_format_projections_sets_source_attr_with_filenames():
    ds = make_projections_dataset(("file", "years", "samples", "locations"))
    result = format_projections(ds)
    assert "file0.nc,file1.nc" in result.attrs["source"]


def test_format_projections_raises_on_nonuniform_year_step():
    ds = make_projections_dataset(
        ("file", "years", "samples", "locations"), year_step_values=(10, 5)
    )
    with pytest.raises(WorkflowTotalerError):
        format_projections(ds)


def test_total_projections_sums_over_file_dim():
    ds = make_projections_dataset(("file", "years", "samples", "locations"))
    formatted = format_projections(ds)
    result = total_projections(formatted, workflow_name="test_workflow")

    assert "file" not in result.dims
    np.testing.assert_array_equal(
        result["sea_level_change"].values,
        formatted["sea_level_change"].sum(dim="file").values,
    )


def test_total_projections_sets_sea_level_change_attrs():
    ds = make_projections_dataset(("file", "years", "samples", "locations"))
    formatted = format_projections(ds)
    result = total_projections(formatted, workflow_name="my_workflow")

    attrs = result["sea_level_change"].attrs
    assert attrs["units"] == "mm"
    assert attrs["workflow_name"] == "my_workflow"
    assert np.isnan(attrs["missing_value"])


def test_total_projections_preserves_top_level_attrs():
    ds = make_projections_dataset(("file", "years", "samples", "locations"))
    formatted = format_projections(ds)
    result = total_projections(formatted, workflow_name="my_workflow")

    assert result.attrs["source"] == formatted.attrs["source"]


def test_get_projections_combines_multiple_files_along_file_dim(tmp_path):
    years = list(range(2020, 2161, 10))
    path1 = write_component_dataset(tmp_path / "file1.nc", years)
    path2 = write_component_dataset(tmp_path / "file2.nc", years)

    result = get_projections(
        paths_list=[str(path1), str(path2)],
        pyear_start=2020,
        pyear_end=2160,
        pyear_step=10,
    )

    assert result.sizes["file"] == 2
    assert sorted(result["file"].values.tolist()) == sorted([str(path1), str(path2)])


def test_get_projections_adds_year_step_coord(tmp_path):
    years = list(range(2020, 2161, 10))
    path = write_component_dataset(tmp_path / "file1.nc", years)

    result = get_projections(
        paths_list=[str(path)],
        pyear_start=2020,
        pyear_end=2160,
        pyear_step=10,
    )

    assert result["year_step"].values.item() == 10


def test_get_projections_subsets_years_when_range_mismatches_pyear_bounds(
    tmp_path, caplog
):
    years = list(range(2010, 2171, 10))
    path = write_component_dataset(tmp_path / "file1.nc", years)

    with caplog.at_level(logging.WARNING):
        result = get_projections(
            paths_list=[str(path)],
            pyear_start=2020,
            pyear_end=2160,
            pyear_step=10,
        )

    assert result["years"].min().item() == 2020
    assert result["years"].max().item() == 2160
    assert any("does not match" in record.message for record in caplog.records)


def test_get_projections_raises_on_nonuniform_step_within_file(tmp_path):
    years = [2020, 2030, 2050]
    path = write_component_dataset(tmp_path / "file1.nc", years)

    with pytest.raises(AssertionError):
        get_projections(
            paths_list=[str(path)],
            pyear_start=2020,
            pyear_end=2050,
            pyear_step=10,
        )


def test_get_projections_normalizes_dim_order_per_file(tmp_path):
    years = list(range(2020, 2161, 10))
    path = write_component_dataset(
        tmp_path / "file1.nc", years, dim_order=("locations", "years", "samples")
    )

    result = get_projections(
        paths_list=[str(path)],
        pyear_start=2020,
        pyear_end=2160,
        pyear_step=10,
    )

    dims = tuple(
        d for d in result["sea_level_change"].dims if d in CANONICAL_COORD_ORDER
    )
    assert dims == CANONICAL_COORD_ORDER
