import xarray as xr
import logging

logger = logging.getLogger(__name__)


def write_totaled_projections(
    outpath: str,
    totaled_ds: xr.Dataset,
) -> None:
    encoding = {"sea_level_change": {"zlib": True, "complevel": 4, "dtype": "f4"}}
    totaled_ds.to_netcdf(outpath, encoding=encoding)
    logger.info("Totaled projections written to %s", outpath)
