# facts-total

This is a minimal prototype of a total module for summing sealevel rise projections generated from different sources and modules. `facts-total` is a CLI tool that accepts a path to each netCDF file you would like summed as well as an output path where the summed result will be written. Each input netCDF file represents output from a FACTS sea level component module. It is the responsibility of the user to ensure that the desired and correct files are specified; check that file paths are correct and that each file specified belongs to the same scale ('global' or 'local'). 

It is possible to run multiple FACTS sea-level components with different default values for common parameters such as `pyear-start` and `pyear-end`. If that happens, `total` will not cause a failure, but will show a message similar to the following: 
```shell
⚠️⚠️ Start years are not the same across all datasets. Check default
values of --pyear-start in these modules. Received: [YYYY YYYY]. ⚠️⚠️
```

## Example running cli
Using output from an 'experiment' run using docker compose that multiple modules. Can specify any number of `--item` inputs, include a path to a module-level output for each. 

Clone repo (`new_pkg` branch):
```shell
git clone --single-branch --branch new_pkg git@github.com:fact-sealevel/facts-total.git
```

From the project root, create a Docker container:
```shell
docker build -t facts-total .
```

Run facts-total in a container. Mount the directory containing the input data as a volume, and the directory where you'd like to write the output file. Then, use the file names in the actual arguments:
```shell
docker run --rm \
-v /path/to/input/data:/mnt/total_in \
-v /path/to/output/data:/mnt/total_out \
facts-total \
--item /mnt/total_in/module_output_1.nc \
--item /mnt/total_in/module_output_2.nc \
--item /mnt/total_in/module_output_3.nc \
--output-path /mnt/total_out/totaled_output.nc
```

## Features 
```shell
Usage: facts-total [OPTIONS]

Options:
  --name TEXT         Name of the workflow being totaled.  [default:
                      my_workflow_name]
  --item TEXT         Paths to component-level projection netcdf files to be
                      totaled.  [required]
  --output-path TEXT  Path to write totaled projections netcdf file.
                      [required]
  --help              Show this message and exit.
```

See the above by running:
```shell
docker run --rm facts-total --help
```

## Results

If this module runs successfully, it will write a NetCDF file to the provided location. The file contains the summmed projections from the provided inputs. 

## Build the container locally
You can build the container with Docker by cloning the repository locally and then running the following command from the repository root:

```shell
docker build -t facts-total .
```
## Support 

Source code is available online at https://github.com/fact-sealevel/facts-total. This software is open source, available under the MIT license.

Please file issues in the issue tracker at https://github.com/fact-sealevel/facts-total/issues.