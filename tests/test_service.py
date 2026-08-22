from facts_total.service import (
    item_tuple_to_list,
)
from hypothesis import given, strategies as st

tuples_of_str = st.lists(
    st.text(alphabet=st.characters(whitelist_categories=("L", "N")), min_size=1),
    min_size=0,
    max_size=20,
).map(tuple)


@given(items=tuples_of_str)
def test_item_tuple_to_list_returns_list(items):
    """items arg from cli is a tuple of str. test that service fn correctly turns into list."""
    result = item_tuple_to_list(items)
    assert isinstance(result, list)
    assert result == list(items)
