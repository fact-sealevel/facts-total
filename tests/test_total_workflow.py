from facts_total.total_workflow import normalize_slc_dim_order
import numpy as np
import xarray as xr


def make_dataset(dim_order):
    """Helper function to create minimal dataset"""

    sizes = {"samples": 100, "years": 14, "locations": 1}
    shape = [sizes[d] for d in dim_order]
    return xr.Dataset(
        {"sea_level_change": xr.DataArray(np.ones(shape), dims=dim_order)},
        coords={
            "samples": np.arange(100),
            "years": np.arange(2020, 2161, 10),
            "locations": np.array([12]),
        },
    )

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
