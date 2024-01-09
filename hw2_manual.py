"""
Cynthia Hong
CSE 163 AF
This program implements of the function of hw2.
The pokemon dataset stores some imagined data about a player’s pokemon.
This program focuses on providing descriptive statistics
for summarizing the pokemon dataset, such as computing the mean
or count of a certain column.
Each function is solved without pandas.
"""


def species_count(data):
    """
    Gives dataset.
    Returns the number of unique pokemon species as determined
    by the name attribute.
    """
    unique_specie = set()
    for monster in data:
        unique_specie.add(monster['name'])
    return len(unique_specie)


def max_level(data):
    """
    Gives dataset and returns a 2-element tuple including
    the highest level and its name of the pokemon.
    If there are many pokemons with the highest level, return tuple of
    the highest level and the name of pokemon that first appears.
    """
    max_level = 0
    monster_name = ""
    for monster in data:
        if monster['level'] > max_level:
            max_level = monster['level']
            monster_name = monster['name']
    return monster_name, max_level


def filter_range(data, lower_bound, upper_bound):
    """
    Gives dataset, lower bound, and upper bound.
    Returns a list of the names of pokemon whose level fall between
    lower bound(inclusive) and upper bound(exclusive),
    as the order names appear in the dataset.
    """
    filter_names = []
    for monster in data:
        if monster['level'] >= lower_bound and monster['level'] < upper_bound:
            filter_names.append(monster['name'])
    return filter_names


def mean_attack_for_type(data, type_name):
    """
    Gives dataset and type_name and returns the mean atk
    for all the Pokemon in the dataset with the given type_name.
    If there are no pokemon of the given type, return None.
    """
    total = 0
    count = 0
    for monster in data:
        if monster['type'] == type_name:
            total += monster['atk']
            count += 1
    if count == 0:
        return None
    else:
        return total / count


def count_types(data):
    """
    Gives dataset and returns a dictionary representing for
    each pokemon type the number of pokemon of that type.
    The order of the keys in the returned dictionary does not matter.
    """
    counts = {}
    for row in data:
        monster = row['type']
        if monster in counts:
            counts[monster] += 1
        else:
            counts[monster] = 1
    return counts


def mean_attack_per_type(data):
    """
    Gives dataset and returns a dictionary representing for
    each pokemon type the mean atk of pokemon of that type.
    The order of the keys in the returned dictionary does not matter.
    """
    counts = {}
    sums = {}
    result = {}
    for row in data:
        monster = row['type']
        if monster in counts:
            counts[monster] += 1
            sums[monster] += row['atk']
        else:
            counts[monster] = 1
            sums[monster] = row['atk']
    for each in counts:
        result[each] = sums[each] / counts[each]
    return result
