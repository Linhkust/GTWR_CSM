'''
This package is divided into two components:

1. SELECTION: select the most similar properties near the target property
with the consideration of spatial and temporal characteristics.

Parameters determined in this component

1.1 Initial Selection
Spatial range: distance threshold;
Temporal range: weeks, months, years, etc.

1.2 Further Selection
Similarity measure criterion: mutual information based Euclidean distance

After the selection process, we can get the similar properties

2. ADJUSTMENT: when the similar properties are identified, their property value should be
integrated into the predicted market value of the target property.

How to integrate: a novel weight function using GTWR

'''
import datetime
import time
import pandas as pd
import requests
import numpy as np
from tqdm import tqdm
from pyproj import Transformer
from geopy.distance import geodesic
from scipy.stats import entropy
import concurrent.futures
from functools import partial
import warnings
from datetime import datetime, timedelta


warnings.filterwarnings('ignore')

'''
Data Processing
Output: final data
'''


def data_processing(data):
    # Delete unwanted column
    unwanted_columns = ['Source.Name', 'District_html', 'Estate_html', 'Contract', 'Cat_ID', 'Arearaw',
                        'Deal_ID']
    df = data.drop(unwanted_columns, axis=1)

    # Transaction date transformation
    df['Date'] = pd.to_datetime(df['Date'])

    # Outliers, missing values, inconsistent values

    # Outliers in price: different units of price (HK$ and million HK$)
    sorted_price = data['Price'].sort_values()  # sort the values as ascending order
    diff = sorted_price.diff()  # calculate the difference
    gap_point = sorted_price.loc[diff.idxmax()]  # find the data gap point
    df['Price'] = df['Price'].apply(lambda x: x / 1e6 if x >= gap_point else x)  # HK$ to Million HK$

    # Inconsistent values: Delete the unit of GFA and SFA
    df['GFA'] = df['GFA'].apply(lambda x: x.replace('ft²', ''))
    df['SFA'] = df['SFA'].apply(lambda x: x.replace('ft²', ''))

    # delete missing values of GFA and SFA
    df = df.drop(df[(df['GFA'] == '--') & (df['SFA'] == '--')].index)
    df['Area'] = df.apply(lambda x: x.GFA if x.GFA != '--' else x.SFA, axis=1)

    df = df.drop(['GFA', 'SFA', 'Change', 'GFA Price', 'SFA Price'], axis=1)

    df.to_csv('data.csv', index=False, encoding='utf_8_sig')


# Feature construction
def spatial_information(data):
    data.replace(np.nan, '', inplace=True)

    for i in tqdm(range(len(data))):
        if data.loc[i, 'Block'] is None:
            address = data.loc[i, 'Estate']
        else:
            address = data.loc[i, 'Estate'] + ' ' + data.loc[i, 'Block']

        # Build the URL for retrieving Easting and Northing
        location_url = "https://geodata.gov.hk/gs/api/v1.0.0/locationSearch?q={}".format(address)
        response = requests.get(location_url)
        response = response.json()

        # Retrieve the x and y information
        x = response[0]['x']
        y = response[0]['y']

        # Add x and y to the dataset
        data.loc[i, 'x'] = x
        data.loc[i, 'y'] = y

        # Server rest
        time.sleep(1)

    data.to_csv('data_xy.csv', index=None)


def facility_variables(row, data, facility, wifi):
    # HKGrid1980 to WGS84
    tf = Transformer.from_crs('epsg:2326', 'epsg:4326', always_xy=True)

    # property data
    data['Longitude'] = data.apply(lambda x: tf.transform(x['x'], x['y'])[0], axis=1)
    data['Latitude'] = data.apply(lambda x: tf.transform(x['x'], x['y'])[1], axis=1)

    # facility data
    facility['Longitude'] = facility.apply(lambda x: tf.transform(x['EASTING'], x['NORTHING'])[0], axis=1)
    facility['Latitude'] = facility.apply(lambda x: tf.transform(x['EASTING'], x['NORTHING'])[1], axis=1)

    # wifi_data
    wifi['Longitude'] = wifi.apply(lambda x: tf.transform(x['Easting'], x['Northing'])[0], axis=1)
    wifi['Latitude'] = wifi.apply(lambda x: tf.transform(x['Easting'], x['Northing'])[1], axis=1)

    # 每个property都要计算一遍
    results = []

    # Initial rectangular selection
    facility_selection = facility[(facility['EASTING'] <= data.loc[row, 'x'] + 1000)
                        & (facility['EASTING'] >= data.loc[row, 'x'] - 1000)
                        & (facility['NORTHING'] >= data.loc[row, 'y'] - 1000)
                        & (facility['NORTHING'] <= data.loc[row, 'y'] + 1000)].reset_index(drop=True)

    facility_selection['distance'] = facility_selection.apply(lambda x: geodesic((data.loc[row, 'Latitude'],
                                                                                  data.loc[row, 'Longitude']),
                                                                                 (x['Latitude'],
                                                                                  x['Longitude'])).m,
                                                                                  axis=1)
    try:
        wifi_selection = wifi[(wifi['Easting'] <= data.loc[row, 'x'] + 1000)
                              & (wifi['Easting'] >= data.loc[row, 'x'] - 1000)
                              & (wifi['Northing'] >= data.loc[row, 'y'] - 1000)
                              & (wifi['Northing'] <= data.loc[row, 'y'] + 1000)].reset_index(drop=True)

        wifi_selection['distance'] = wifi_selection.apply(lambda x: geodesic((data.loc[row, 'Latitude'],
                                                                          data.loc[row, 'Longitude']),
                                                                         (x['Latitude'],
                                                                          x['Longitude'])).m,
                                                                          axis=1)
        wifi_1km = wifi_selection[wifi_selection['distance'] <= 1000].reset_index(drop=True)
        wifi_density = len(wifi_1km)

    except (KeyError, ValueError):
        wifi_density = 0

    facilities_1km = facility_selection[facility_selection['distance'] <= 1000][['GEONAMEID', 'CLASS', 'TYPE', 'distance']].reset_index(drop=True)

    variables = {}

    # POI density
    poi_density = len(facilities_1km)
    variables['wifi_hk'] = wifi_density
    variables['POI_density'] = poi_density

    # POI diversity
    # Number of CLASS and TYPE
    num_class = len(facilities_1km['CLASS'].unique())
    num_type = len(facilities_1km['TYPE'].unique())
    variables['Num_class'] = num_class
    variables['Num_type'] = num_type

    # Entropy-based CLASS diversity
    class_unique_num = facilities_1km['CLASS'].value_counts()
    class_unique_percentage = class_unique_num / class_unique_num.sum()
    class_unique_percentage = class_unique_percentage.tolist()
    class_entropy = entropy(class_unique_percentage, base=2)  # CLASS_Entropy
    variables['Class_diversity'] = class_entropy

    # Entropy-based TYPE diversity
    type_unique_num = facilities_1km['TYPE'].value_counts()
    type_unique_percentage = type_unique_num / type_unique_num.sum()
    type_unique_percentage = type_unique_percentage.tolist()
    type_entropy = entropy(type_unique_percentage, base=2)  # TYPE_Entropy
    variables['Type_diversity'] = type_entropy

    # Distance to the nearest unique TYPE of facility
    facility_type = facilities_1km['TYPE'].unique()
    for j in range(len(facility_type)):
        distance = facilities_1km[facilities_1km['TYPE'] == facility_type[j]]['distance'].min()
        variables[facility_type[j]] = distance
    return variables


def parallel_computing(worker, data, facility, wifi):
    # Assign the core missions

    iter_num = len(data) // worker
    fun_partial = partial(facility_variables, data=data, facility=facility, wifi=wifi)

    info = []
    for i in tqdm(range(iter_num+1)):

        if (i+1)*worker <= len(data):
            iter_term = [i*worker+num for num in range(worker)]
        else:
            iter_term = [num for num in range(i*worker, len(data))]

        # Construct the variables
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker) as executor:
            results = executor.map(fun_partial, iter_term)
            for result in results:
                info.append(result)

    tf = Transformer.from_crs('epsg:2326', 'epsg:4326', always_xy=True)
    data['Longitude'] = data.apply(lambda x: tf.transform(x['x'], x['y'])[0], axis=1)
    data['Latitude'] = data.apply(lambda x: tf.transform(x['x'], x['y'])[1], axis=1)

    df = pd.concat([pd.DataFrame(l, index=[0]) for l in info], axis=0, ignore_index=True)
    df = pd.concat([data, df], axis=1)
    df.to_csv('data_variables.csv', index=False)


'''
Variable Creation and Algorithm Design
'''


def gtwr_knn(data, threshold, spatial, temporal):
    # variable creation: select the independent and dependent variables
    # This is mainly about the POI related variables

    # POIs that significantly affect the housing prices
    selected_poi = ['MAL', 'SMK', 'KDG', 'PRS', 'SES', 'PAR', 'PLG', 'RGD', 'BUS', 'MIN', 'CPO', 'MTA']
    for i, poi in enumerate(selected_poi):
        data[poi+'_Walk'] = data.apply(lambda x: 1 if x[poi] <= threshold else 0, axis=1)

    # define the x and y variables
    x_variables = ['Date',
                   'Floor',
                   'Area',
                   'x',
                   'y',
                   'wifi_hk',
                   'POI_density',
                   'Num_class',
                   'Num_type',
                   'Class_diversity',
                   'Type_diversity'] + [j+'_Walk' for j in selected_poi]
    y_variable = ['Price']

    x = data[x_variables]
    y = data[y_variable]

    # Initial selection
    for i in range(len(data)):
        # Use drop function to obtain all properties except the target one
        other_properties = data.drop(i)

        # Spatial_proximity
        rectangle_properties = other_properties[(other_properties['x'] <= data.loc[i, 'x'] + spatial) &
                                                (other_properties['x'] >= data.loc[i, 'x'] - spatial)&
                                                (other_properties['y'] <= data.loc[i, 'y'] + spatial)&
                                                (other_properties['y'] >= data.loc[i, 'y'] - spatial)]

        rectangle_properties['distance'] = rectangle_properties.apply(lambda x: geodesic((data.loc[i, 'Latitude'],
                                                                      data.loc[i, 'Longitude']),
                                                                      (x['Latitude'],
                                                                      x['Longitude'])).m,
                                                                      axis=1)

        spatial_properties = rectangle_properties[rectangle_properties['distance'] <= spatial]

        # Temporal_proximity
        spatial_properties['Date'] = pd.to_datetime(spatial_properties['Date'], format='%d/%m/%Y')

        # Temporal gap
        delta = timedelta(days=temporal)

        transaction_date = datetime.strptime(data.loc[i, 'Date'], '%d/%m/%Y')
        start_time = transaction_date - delta

        # Select the properties within the selected time period
        spatial_temporal_properties = spatial_properties[(spatial_properties['Date'] >= start_time) &
                                                         (spatial_properties['Date'] <= transaction_date)].reset_index(drop=True)
        spatial_temporal_properties['time'] = spatial_temporal_properties.apply(lambda x: (transaction_date-x['Date']).total_seconds()/86400, axis=1)
        print(spatial_temporal_properties)
        break


if __name__ == "__main__":
    # data import
    # property_data = pd.read_csv('data_xy.csv', encoding='unicode_escape')
    # facility_data = pd.read_csv('GeoCom4.0_202203.csv', low_memory=False)
    # wifi_data = pd.read_csv('WIFI_HK.csv', low_memory=False)

    # processor = 8
    #
    # parallel_computing(processor, property_data, facility_data, wifi_data)

    # GTWR-based algorithm design
    walk_threshold = 400
    spatial_bandwidth = 500
    temporal_bandwidth = 30
    data = pd.read_csv('data_variables.csv', encoding='unicode_escape')
    gtwr_knn(data, walk_threshold, spatial_bandwidth, temporal_bandwidth)





