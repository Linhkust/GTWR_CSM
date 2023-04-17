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
from sklearn.neighbors import KDTree
import multiprocessing as mp

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


def distance_facility(data, facility):
    # HKGrid1980 to WGS84
    tf = Transformer.from_crs('epsg:2326', 'epsg:4326', always_xy=True)

    # property data
    data['Longitude'] = data.apply(lambda x: tf.transform(x['x'], x['y'])[0], axis=1)
    data['Latitude'] = data.apply(lambda x: tf.transform(x['x'], x['y'])[1], axis=1)

    # facility data
    facility['Longitude'] = facility.apply(lambda x: tf.transform(x['EASTING'], x['NORTHING'])[0], axis=1)
    facility['Latitude'] = facility.apply(lambda x: tf.transform(x['EASTING'], x['NORTHING'])[1], axis=1)

    temp = []
    for i in range(len(data)):
        facility['distance'] = facility.apply(lambda x: geodesic((data.loc[i, 'Latitude'], data.loc[i, 'Longitude']),
                                                                 (x['Latitude'], x['Longitude'])).m, axis=1)
        facilities_1km = facility[facility['distance'] <= 1000][['GEONAMEID', 'distance']].reset_index(drop=True)
        facilities_1km.insert(loc=0, column='ID', value=data.loc[i, 'ID'])
        temp.append(facilities_1km)


if __name__ == "__main__":
    # data = pd.read_csv('final.csv')
    # data_processing(data)
    new_data = pd.read_csv('data.csv')
    test_data = pd.read_csv('data_xy.csv')
    facility_data = pd.read_csv('GeoCom4.0_202203.csv', low_memory=False)
    distance_facility(test_data, facility_data)

