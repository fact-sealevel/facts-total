import click
from facts_total.total_workflow import WorkflowTotaler


@click.command()
@click.option(
    "--name",
    type=str,
    default="my_workflow_name",
    show_default=True,
    help="Name of the workflow being totaled.",
)
@click.option(
    "--item",
    multiple=True,
    required=True,
    help="Paths to component-level projection netcdf files to be totaled.",
)
@click.option(
    "--pyear-start",
    type=int,
    required=True,
    help="Enter the pyear-start value used for the individual modules. If modules used different pyear-start values, enter the one you would like used for the totaled output.",
)
@click.option(
    "--pyear-end",
    type=int,
    required=True,
    help="Enter the pyear-end value used for the individual modules. If modules used different pyear-end values, enter the one you would like used for the totaled output.",
)
@click.option(
    "--pyear-step",
    type=int,
    required=True,
    help="Enter the pyear-step value used for the individual modules. If modules used different pyear-step values, enter the one you would like used for the totaled output.",
)
@click.option(
    "--output-path",
    type=str,
    required=True,
    help="Path to write totaled projections netcdf file.",
)
def main(name, item, output_path, pyear_start, pyear_end, pyear_step):
    click.echo("Hello from FACTS totaling!")

    # Make list of input paths
    paths_list = list(item)

    # Create totaler obj
    totaler = WorkflowTotaler(
        name=name,
        paths_list=paths_list,
        pyear_start=pyear_start,
        pyear_end=pyear_end,
        pyear_step=pyear_step,
    )

    # Read files and total projections
    totaler.get_projections()

    # Calc sum
    totaler.total_projections()

    # Write totaled projections to file
    totaler.write_totaled_projections(outpath=output_path)
