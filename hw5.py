"""
Cynthia Hong
CSE 163 AF
This file is for putting my implementations of HW5.
The program is to analyze and plot geospatial data to
investigate food deserts in Washington.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


def load_in_data(shp_file_name, csv_file_name):
    """
    Takes two parameters, the filename for the census dataset
    and the filename for the food access dataset.
    Merge the two datasets and
    return the result as a GeoDataFrame.
    Assume the census identifier column names exist,
    but don’t assume any other columns in the datasets.
    """
    census = gpd.read_file(shp_file_name)
    food_access = pd.read_csv(csv_file_name)
    geo_df = census.merge(food_access, left_on='CTIDFP00',
                          right_on='CensusTract', how='left')
    return geo_df


def percentage_food_data(state_data):
    """
    Takes the merged data and returns the percentage of census tracts
    in Washington for which we have food access data.
    The percentage should be a float between 0 and 100.
    """
    state = state_data[(state_data['State'] == "WA")]
    return len(state) * 100 / len(state_data)


def plot_map(state_data):
    """
    Takes the merged data and plots the shapes of all the census
    tracts in Washington in a file map.png.
    """
    state_data.plot()
    plt.title('Washington State')
    plt.savefig('map.png')


def plot_population_map(state_data):
    """
    Takes the merged data and plots the shapes of all the census tracts in
    Washington in a file population_map.png where each census tract is colored
    according to population. There will be some missing census tracts.
    """
    fig, ax = plt.subplots(1)
    state_data.plot(ax=ax, color='#EEEEEE')
    state_data.plot(ax=ax, column='POP2010', legend=True)
    plt.title('Washington Census Tract Populations')
    plt.savefig('population_map.png')


def plot_population_county_map(state_data):
    """
    Takes the merged data and plots the shapes of all the census tracts in
    Washington in a file county_population_map.png where each county is colored
    according to population. There will be some missing counties.
    """
    populations = state_data[['County', 'POP2010', 'geometry']]
    population_by_country = populations.dissolve(by='County', aggfunc='sum')
    fig, ax = plt.subplots(1)
    state_data.plot(ax=ax, color='#EEEEEE')
    population_by_country.plot(ax=ax, column='POP2010', legend=True)
    plt.title('Washington County Populations')
    plt.savefig('county_population_map.png')


def plot_food_access_by_county(state_data):
    """
    Takes the merged data and produces 4 plots on the same figure showing
    information about food access across income level.
    """
    df = state_data[['County', 'geometry', 'POP2010', 'lapophalf', 'lapop10',
                     'lalowihalf', 'lalowi10']]
    df = df.dissolve(by='County', aggfunc='sum')
    df['lapophalf_ratio'] = df['lapophalf'] / df['POP2010']
    df['lapop10_ratio'] = df['lapop10'] / df['POP2010']
    df['lalowihalf_ratio'] = df['lalowihalf'] / df['POP2010']
    df['lalowi10_ratio'] = df['lalowi10'] / df['POP2010']
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 10))
    # The first
    state_data.plot(ax=ax1, color='#EEEEEE')
    df.plot(ax=ax1, column='lapophalf_ratio', vmin=0, vmax=1, legend=True)
    ax1.set_title('Low Access: Half')
    # The second
    state_data.plot(ax=ax2, color='#EEEEEE')
    df.plot(ax=ax2, column='lalowihalf_ratio', vmin=0, vmax=1, legend=True)
    ax2.set_title('Low Access + Low Income: Half')
    # The third
    state_data.plot(ax=ax3, color='#EEEEEE')
    df.plot(ax=ax3, column='lapop10_ratio', vmin=0, vmax=1, legend=True)
    ax3.set_title('Low Access: 10')
    # The forth
    state_data.plot(ax=ax4, color='#EEEEEE')
    df.plot(ax=ax4, column='lalowi10_ratio', vmin=0, vmax=1, legend=True)
    ax4.set_title('Low Access + Low Income: 10')
    plt.savefig('county_food_access.png')


def plot_low_access_tracts(state_data):
    """
    Takes the merged data and plots all census tracts considered “low access”
    in a file low_access.png.
    """
    urban = state_data[state_data['Urban'] == 1]
    rural = state_data[state_data['Rural'] == 1]
    urban_low = urban[((urban['lapophalf'] / urban['POP2010']) >= 0.33)
                      | (urban['lapophalf'] >= 500)]
    rural_low = rural[((rural['lapop10'] / rural['POP2010']) >= 0.33)
                      | (rural['lapop10'] >= 500)]
    fig, ax = plt.subplots(1)
    state_data.plot(ax=ax, color='#EEEEEE')
    state_data.dropna().plot(ax=ax, color='#AAAAAA')
    urban_low.plot(ax=ax)
    rural_low.plot(ax=ax)
    plt.title('Low Access Census Tracts')
    plt.savefig('low_access.png')


def main():
    state_data = load_in_data(
        '/course/food_access/tl_2010_53_tract00/tl_2010_53_tract00.shp',
        '/course/food_access/food_access.csv'
    )
    print(percentage_food_data(state_data))
    plot_map(state_data)
    plot_population_map(state_data)
    plot_population_county_map(state_data)
    plot_food_access_by_county(state_data)
    plot_low_access_tracts(state_data)


if __name__ == '__main__':
    main()
