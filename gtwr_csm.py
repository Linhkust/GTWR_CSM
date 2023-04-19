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

warnings.filterwarnings('ignore')

'''
Data Processing
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


def distance_facility(data, facility, wifi, threshold):
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
    for i in tqdm(range(len(data))):

        # Initial rectangular selection
        facility_selection = facility[(facility['EASTING'] <= data.loc[i, 'x'] + 1000)
                            & (facility['EASTING'] >= data.loc[i, 'x'] - 1000)
                            & (facility['NORTHING'] >= data.loc[i, 'y'] - 1000)
                            & (facility['NORTHING'] <= data.loc[i, 'y'] + 1000)].reset_index(drop=True)

        facility_selection['distance'] = facility_selection.apply(lambda x: geodesic((data.loc[i, 'Latitude'],
                                                                                      data.loc[i, 'Longitude']),
                                                                                     (x['Latitude'],
                                                                                      x['Longitude'])).m,
                                                                                      axis=1)
        try:
            wifi_selection = wifi[(wifi['Easting'] <= data.loc[i, 'x'] + 1000)
                                  & (wifi['Easting'] >= data.loc[i, 'x'] - 1000)
                                  & (wifi['Northing'] >= data.loc[i, 'y'] - 1000)
                                  & (wifi['Northing'] <= data.loc[i, 'y'] + 1000)].reset_index(drop=True)

            wifi_selection['distance'] = wifi_selection.apply(lambda x: geodesic((data.loc[i, 'Latitude'],
                                                                              data.loc[i, 'Longitude']),
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
            variables[facility_type[j] + '_walk'] = 1 if distance < threshold else 0

        results.append(variables)

    df = pd.concat([pd.DataFrame(l, index=[0]) for l in results], axis=0, ignore_index=True)
    df = pd.concat([data, df], axis=1)

    df.to_csv('data_variables.csv', index=False)


if __name__ == "__main__":
    # data import
    property_data = pd.read_csv('data_xy.csv', encoding='unicode_escape')
    facility_data = pd.read_csv('GeoCom4.0_202203.csv', low_memory=False)
    wifi_data = pd.read_csv('WIFI_HK.csv', low_memory=False)
    walk_threshold = 400

    # Construct the variables
    distance_facility(property_data, facility_data, wifi_data, walk_threshold)





