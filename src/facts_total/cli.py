import click
from facts_total.service import (
    run_facts_total,
)
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@click.command()
@click.option(
    "--name",
    type=str,
    default="my_workflow_name",
    show_default=True,
    required=True,
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
    click.echo("Hello from FACTS totaling (testing)!")

    run_facts_total(
        name=name,
        item=item,
        output_path=output_path,
        pyear_start=pyear_start,
        pyear_end=pyear_end,
        pyear_step=pyear_step,
    )
