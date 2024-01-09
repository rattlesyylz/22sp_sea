"""
Cynthia Hong
CSE 163 AF
This program implements of the function of hw2.
The pokemon dataset stores some imagined data about a player’s pokemon.
This program focuses on providing descriptive statistics for summarizing
the pokemon dataset, such as computing the mean or count of a certain column.
Each function is solved with pandas.
"""


def species_count(data):
    """
    Takes a Pandas DataFrame.
    Returns the number of unique pokemon species as determined
    by the name attribute.
    """
    return len(data['name'].unique())


def max_level(data):
    """
    Takes a Pandas DataFrame.
    Returns a 2-element tuple including the highest level
    and its name of the pokemon.
    If there are many pokemons with the highest level, return tuple of
    the highest level and the name of pokemon that first appears.
    """
    max_id = data['level'].idxmax()
    element = data.loc[max_id, ['name', 'level']]
    return tuple(element)


def filter_range(data, lower_bound, upper_bound):
    """
    Takes a Pandas DataFrame and gives lower bound, and upper bound.
    Returns a list of the names of pokemon whose level fall between
    lower bound(inclusive) and upper bound(exclusive),
    as the order names appear in the dataset.
    """
    bound = data[(data['level'] >= lower_bound) &
                 (data['level'] < upper_bound)]
    return list(bound['name'])


def mean_attack_for_type(data, type_name):
    """
    Takes a Pandas DataFrame and gives type_name.
    Returns the mean atk for all the Pokemon in the dataset
    with the given type_name.
    If there are no pokemon of the given type, return None.
    """
    type_df = data[data['type'] == type_name]
    if len(type_df) == 0:
        return None
    else:
        return type_df['atk'].mean()


def count_types(data):
    """
    Takes a Pandas DataFrame.
    Returns a dictionary representing for each pokemon type
    the number of pokemon of that type.
    The order of the keys in the returned dictionary does not matter.
    """
    return dict(data.groupby('type')['type'].count())


def mean_attack_per_type(data):
    """
    Takes a Pandas DataFrame.
    Returns a dictionary representing for each pokemon type
    the mean atk of pokemon of that type.
    The order of the keys in the returned dictionary does not matter.
    """
    return dict(data.groupby('type')['atk'].mean())
