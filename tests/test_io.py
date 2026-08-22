import numpy as np
import xarray as xr

from facts_total.core import format_projections, total_projections
from facts_total.io import write_totaled_projections

from test_core import make_projections_dataset


def make_totaled_dataset():
    ds = make_projections_dataset(("file", "years", "samples", "locations"))
    formatted = format_projections(ds)
    return total_projections(formatted, workflow_name="test_workflow")


def test_write_totaled_projections_creates_file(tmp_path):
    outpath = tmp_path / "totaled.nc"
    totaled_ds = make_totaled_dataset()

    write_totaled_projections(str(outpath), totaled_ds)

    assert outpath.exists()


def test_write_totaled_projections_returns_none(tmp_path):
    outpath = tmp_path / "totaled.nc"
    totaled_ds = make_totaled_dataset()

    result = write_totaled_projections(str(outpath), totaled_ds)

    assert result is None


def test_write_totaled_projections_data_roundtrips(tmp_path):
    outpath = tmp_path / "totaled.nc"
    totaled_ds = make_totaled_dataset()

    write_totaled_projections(str(outpath), totaled_ds)

    with xr.open_dataset(outpath) as reread_ds:
        np.testing.assert_allclose(
            reread_ds["sea_level_change"].values,
            totaled_ds["sea_level_change"].values,
            rtol=1e-6,
        )


def test_write_totaled_projections_stores_sea_level_change_as_float32(tmp_path):
    outpath = tmp_path / "totaled.nc"
    totaled_ds = make_totaled_dataset()

    write_totaled_projections(str(outpath), totaled_ds)

    with xr.open_dataset(outpath) as reread_ds:
        assert reread_ds["sea_level_change"].dtype == np.float32


def test_write_totaled_projections_applies_compression_encoding(tmp_path):
    outpath = tmp_path / "totaled.nc"
    totaled_ds = make_totaled_dataset()

    write_totaled_projections(str(outpath), totaled_ds)

    with xr.open_dataset(outpath) as reread_ds:
        encoding = reread_ds["sea_level_change"].encoding
        assert encoding["zlib"] is True
        assert encoding["complevel"] == 4


def test_write_totaled_projections_preserves_attrs(tmp_path):
    outpath = tmp_path / "totaled.nc"
    totaled_ds = make_totaled_dataset()

    write_totaled_projections(str(outpath), totaled_ds)

    with xr.open_dataset(outpath) as reread_ds:
        assert reread_ds.attrs["source"] == totaled_ds.attrs["source"]
        assert (
            reread_ds["sea_level_change"].attrs["workflow_name"]
            == totaled_ds["sea_level_change"].attrs["workflow_name"]
        )
        assert (
            reread_ds["sea_level_change"].attrs["units"]
            == totaled_ds["sea_level_change"].attrs["units"]
        )
