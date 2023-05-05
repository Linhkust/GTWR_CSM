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
from sklearn.model_selection import train_test_split
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


np.set_printoptions(suppress=True)
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
    data.dropna(subset=['Floor'], inplace=True)

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


def knn_benchmark(data, threshold):
    data.dropna(subset=['Floor'], inplace=True)

    # POIs that significantly affect the housing prices
    selected_poi = ['MAL', 'SMK', 'KDG', 'PRS', 'SES', 'PAR', 'PLG', 'RGD', 'BUS', 'MIN', 'CPO', 'MTA']
    for i, poi in enumerate(selected_poi):
        data[poi + '_Walk'] = data.apply(lambda x: 1 if x[poi] <= threshold else 0, axis=1)

    # define the x and y variables
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day

    x_variables = ['Month',
                   'Day',
                   'Floor',
                   'Area',
                   'x',
                   'y',
                   'wifi_hk',
                   'POI_density',
                   'Num_class',
                   'Num_type',
                   'Class_diversity',
                   'Type_diversity'] + [j + '_Walk' for j in selected_poi]
    y_variable = ['Price']

    train_test_data = pd.concat([data[x_variables], data[y_variable]], axis=1)

    x = train_test_data[x_variables]
    y = train_test_data[y_variable]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    model = KNeighborsRegressor(n_neighbors=5, weights='uniform')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    result = r2_score(y_test, y_pred)
    print(result)


def data_split(data, threshold):
    # variable creation: select the independent and dependent variables
    # This is mainly about the POI related variables

    # POIs that significantly affect the housing prices
    selected_poi = ['MAL', 'SMK', 'KDG', 'PRS', 'SES', 'PAR', 'PLG', 'RGD', 'BUS', 'MIN', 'CPO', 'MTA']
    for i, poi in enumerate(selected_poi):
        data[poi + '_Walk'] = data.apply(lambda x: 1 if x[poi] <= threshold else 0, axis=1)

    # define the x and y variables
    x_variables = ['Date',
                   'Latitude',
                   'Longitude',
                   'Floor',
                   'Area',
                   'x',
                   'y',
                   'wifi_hk',
                   'POI_density',
                   'Num_class',
                   'Num_type',
                   'Class_diversity',
                   'Type_diversity'] + [j + '_Walk' for j in selected_poi]

    y_variable = ['Price']
    train_test_data = pd.concat([data[x_variables], data[y_variable]], axis=1)

    # Standardize the data
    scaled_train_test_data = pd.DataFrame(StandardScaler().fit_transform(train_test_data.iloc[:, 3:]),
                                          columns=train_test_data.columns[3:])

    train_test_data = pd.concat([train_test_data.iloc[:, :3], scaled_train_test_data], axis=1)

    x = train_test_data[x_variables]
    y = train_test_data[y_variable]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    train = pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis=1)

    return train, test


def gtwr_knn(train, spatial, temporal, weight_function, n_neighbors):

    """
    Training process
    """
    pred = []
    test = []
    train = train.reset_index(drop=True)
    train['Date'] = pd.to_datetime(train['Date'], format='%d/%m/%Y')
    for i in tqdm(range(10)):
        """
        SELECTION 选择
        """

        '''Initial selection 选择时空相近的个体'''
        # target property
        target_property = train.iloc[i, :]

        # Other properties
        other_properties = train.drop(i)

        # Spatial_proximity
        rectangle_properties = other_properties[(other_properties['x'] <= target_property['x'] + spatial) &
                                                (other_properties['x'] >= target_property['x'] - spatial) &
                                                (other_properties['y'] <= target_property['y'] + spatial) &
                                                (other_properties['y'] >= target_property['y'] - spatial)]

        rectangle_properties['distance'] = rectangle_properties.apply(lambda x: geodesic((target_property['Latitude'],
                                                                      target_property['Longitude']),
                                                                      (x['Latitude'],
                                                                       x['Longitude'])).m,
                                                                      axis=1)

        spatial_properties = rectangle_properties[rectangle_properties['distance'] <= spatial]

        # Temporal_proximity
        spatial_properties['Date'] = pd.to_datetime(spatial_properties['Date'], format='%d/%m/%Y')

        # Temporal gap: needs to be larger to cover more data
        delta = timedelta(days=temporal)

        transaction_date = target_property['Date']
        start_time = transaction_date - delta

        # Select the properties within the selected time period
        try:
            spatial_temporal_properties = spatial_properties[(spatial_properties['Date'] >= start_time) &
                                                             (spatial_properties['Date'] <= transaction_date)
                                                             ].reset_index(drop=True)

            spatial_temporal_properties['time'] = spatial_temporal_properties.apply(
                lambda x: (transaction_date-x['Date']).total_seconds()/86400, axis=1)

        except ValueError:
            spatial_temporal_properties = spatial_properties

        '''KNN selection 选择属性相似的个体'''
        # Attribute similarity calculation
        initial_selection = spatial_temporal_properties.iloc[:, 3:]
        sub_result = initial_selection.sub(target_property[3:])

        # Attribute distance calculation (Euclidean distance or Cosine similarity)
        initial_selection['attribute_distance'] = sub_result.apply(lambda x: np.sqrt(np.sum(x**2)), axis=1)

        if len(initial_selection) > n_neighbors:
            # Select the most similar properties
            initial_selection_neighbors = initial_selection.nsmallest(n_neighbors, 'attribute_distance')
        else:
            initial_selection_neighbors = initial_selection

        '''
        ADJUSTMENT 
        对于选择得到的个体表现进行预测
        '''

        '''GTWR Weight Creation 创建GTWR权重函数'''

        # Gaussian weight function
        if weight_function == "Gaussian":
            try:
                # Spatial Weight Calculation
                initial_selection_neighbors['spatial_weight'] = initial_selection_neighbors.apply(
                    lambda x: math.exp(-np.square(x['distance'] / spatial_bandwidth)), axis=1)

                # Temporal Weight Calculation
                initial_selection_neighbors['temporal_weight'] = initial_selection_neighbors.apply(
                    lambda x: math.exp(-np.square(x['time'] / temporal_bandwidth)), axis=1)

                # GTWR Weight Calculation
                initial_selection_neighbors['GTWR_weight'] = initial_selection_neighbors.apply(
                    lambda x: x['spatial_weight'] * x['temporal_weight'], axis=1)

            except KeyError:
                # Spatial Weight Calculation
                initial_selection_neighbors['spatial_weight'] = initial_selection_neighbors.apply(
                    lambda x: math.exp(-np.square(x['distance'] / spatial_bandwidth)), axis=1)

                # GTWR Weight Calculation
                initial_selection_neighbors['GTWR_weight'] = initial_selection_neighbors.apply(
                    lambda x: x['spatial_weight'], axis=1)

        # Bi-Square Weight Function
        elif weight_function == "Bi-Square":
            try:
                # Spatial Weight Calculation
                initial_selection_neighbors['spatial_weight'] = initial_selection_neighbors.apply(
                    lambda x: np.square(1 - math.exp(np.square(x['distance']) / np.square(spatial_bandwidth))), axis=1)

                # Temporal Weight Calculation
                initial_selection_neighbors['temporal_weight'] = initial_selection_neighbors.apply(
                    lambda x: np.square(1 - math.exp(np.square(x['time']) / np.square(temporal_bandwidth))), axis=1)

                # GTWR Weight Calculation
                initial_selection_neighbors['GTWR_weight'] = initial_selection_neighbors.apply(
                    lambda x: x['spatial_weight'] * x['temporal_weight'], axis=1)

            except KeyError:
                # Spatial Weight Calculation
                initial_selection_neighbors['spatial_weight'] = initial_selection_neighbors.apply(
                    lambda x: np.square(1 - math.exp(np.square(x['distance']) / np.square(spatial_bandwidth))), axis=1)

                # GTWR Weight Calculation
                initial_selection_neighbors['GTWR_weight'] = initial_selection_neighbors.apply(
                    lambda x: x['spatial_weight'], axis=1)

        # Standardization of the GTWR weights
        # MinMax Scaler
        standardized_data = MinMaxScaler().fit_transform(initial_selection_neighbors['GTWR_weight'].values.reshape(-1, 1))
        initial_selection_neighbors['GTWR_weight_standard'] = standardized_data

        ''' 
        Value aggregation 整合选择得到的个体表现得到预测值
        '''
        weight = initial_selection_neighbors.loc[:, 'GTWR_weight_standard'].reset_index(drop=True)
        price = initial_selection_neighbors.loc[:, 'Price'].reset_index(drop=True)
        predicted_price = weight.dot(price)
        pred.append(predicted_price)
        test.append(target_property['Price'])

    # Training results
    training_results ={'MAE': mean_absolute_error(test, pred),
                       'RMSE': math.sqrt(mean_squared_error(test, pred))
                      }
    print(training_results)


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
    spatial_bandwidth = 1000
    temporal_bandwidth = 15
    n_neighbors = 12

    data = pd.read_csv('data_variables.csv', encoding='unicode_escape')
    # knn_benchmark(data, walk_threshold)
    train_data = data_split(data, walk_threshold)[0]

    gtwr_knn(train_data, spatial_bandwidth, temporal_bandwidth, 'Bi-Square', n_neighbors=n_neighbors)




