from facts_total.core import (
    get_projections,
    format_projections,
    total_projections,
)
from facts_total.io import (
    write_totaled_projections,
)


def run_facts_total(
    name,
    item,
    output_path,
    pyear_start,
    pyear_end,
    pyear_step,
) -> None:
    # Make list of input paths
    paths_list = list(item)

    # Get ds of combined projections
    projections_ds = get_projections(
        paths_list=paths_list,
        pyear_start=pyear_start,
        pyear_end=pyear_end,
        pyear_step=pyear_step,
    )
    # format projections
    formatted_projections = format_projections(combined_ds=projections_ds)

    # Total projections
    totaled_projections = total_projections(formatted_projections, workflow_name=name)

    # write projections
    write_totaled_projections(outpath=output_path, totaled_ds=totaled_projections)
