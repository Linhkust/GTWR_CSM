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
import multiprocessing
import time
import pandas as pd
import requests
import numpy as np
from tqdm import tqdm
from pyproj import Transformer
from geopy.distance import geodesic
from scipy.stats import entropy
import warnings
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import sys

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

    # Generated Features
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


# Parallel operation for feature generation
def parallel_feature_generation(data, facility, wifi):
    # Testing
    data_count = len(data)
    pbar = tqdm(total=data_count, file=sys.stdout, colour='white')
    pbar.set_description('Feature Generation')
    update = lambda *args: pbar.update()

    # VERY IMPORTANT: check how many cores in your PC
    pool = multiprocessing.Pool(processes=12)

    # 定义一个列表来存储每次循环的结果
    results = []

    # 并行运行for循环
    for num in range(data_count):
        # 将任务提交给进程池
        result = pool.apply_async(facility_variables,
                                  args=(num,
                                        data,
                                        facility,
                                        wifi),
                                  callback=update)
        results.append(result)

    # 等待所有进程完成
    pool.close()
    pool.join()

    pred_results = []
    # 打印每次循环的结果
    for result in results:
        pred_results.append(result.get())

    pred_results = pd.DataFrame(pred_results)

    # HKGrid1980 to WGS84
    tf = Transformer.from_crs('epsg:2326', 'epsg:4326', always_xy=True)

    # Add geographic coordinates
    data['Longitude'] = data.apply(lambda x: tf.transform(x['x'], x['y'])[0], axis=1)
    data['Latitude'] = data.apply(lambda x: tf.transform(x['x'], x['y'])[1], axis=1)

    # Combine property information and facility information
    final_data = pd.concat([data, pred_results], axis=1)
    final_data.to_csv('data_features.csv', index=False)


'''
Variable Creation and Algorithm Design
'''


def knn_benchmark(data, threshold, n_neighbors):
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
    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='uniform')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    result1 = mean_absolute_error(y_test[:100], y_pred[:100])
    result2 = math.sqrt(mean_squared_error(y_test[:100], y_pred[:100]))
    result = {'MAE': "%.2f" % result1, 'RMSE': "%.2f" % result2}
    print(result)
    return result


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
                   'x',
                   'y',
                   'Floor',
                   'Area',
                   'wifi_hk',
                   'POI_density',
                   'Num_class',
                   'Num_type',
                   'Class_diversity',
                   'Type_diversity'] + [j + '_Walk' for j in selected_poi]

    y_variable = ['Price']
    train_test_data = pd.concat([data[x_variables], data[y_variable]], axis=1)

    # Standardize the data
    scaled_train_test_data = pd.DataFrame(StandardScaler().fit_transform(train_test_data.iloc[:, 5:]),
                                          columns=train_test_data.columns[5:])

    train_test_data = pd.concat([train_test_data.iloc[:, :5], scaled_train_test_data], axis=1)

    x = train_test_data[x_variables]
    y = train_test_data[y_variable]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    train = pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis=1)

    return train, test


def model_performance(test, pred):
    # Training results
    training_results ={'MAE': "%.2f" % mean_absolute_error(test, pred),
                       'RMSE': "%.2f" % math.sqrt(mean_squared_error(test, pred))
                      }
    return training_results


# Train the data to find the best parameters
def gtwr_knn_training(train, num, spatial, temporal, gtwr_weight, n_neighbors, knn_weight):

    global predicted_price
    train = train.reset_index(drop=True)
    train['Date'] = pd.to_datetime(train['Date'], format='%d/%m/%Y')

    """
    SELECTION 选择
    """

    '''Initial selection 选择时空相近的个体'''

    # target property
    target_property = train.iloc[num, :]

    # Other properties
    other_properties = train.drop(num)

    # Spatial_proximity
    rectangle_properties = other_properties[(other_properties['x'] <= target_property['x'] + spatial) &
                                            (other_properties['x'] >= target_property['x'] - spatial) &
                                            (other_properties['y'] <= target_property['y'] + spatial) &
                                            (other_properties['y'] >= target_property['y'] - spatial)]

    if len(rectangle_properties) > 0:
        rectangle_properties['distance'] = rectangle_properties.apply(lambda x: geodesic((
                                                                                      target_property['Latitude'],
                                                                                      target_property['Longitude']),
                                                                                     (x['Latitude'],
                                                                                      x['Longitude'])).m,
                                                                                      axis=1)
        # Select properties within the distance threshold
        spatial_properties = rectangle_properties[rectangle_properties['distance'] <= spatial]

        if len(spatial_properties) > 0:

            # Temporal gap: needs to be larger to cover more data
            delta = timedelta(days=temporal)

            transaction_date = target_property['Date']
            start_time = transaction_date - delta

            # Select the properties within the selected time period
            # If there is no record with specific period, then choose the data with spatial threshold

            spatial_temporal_properties = spatial_properties[(spatial_properties['Date'] >= start_time) &
                                                                 (spatial_properties['Date'] <= transaction_date)
                                                                 ].reset_index(drop=True)

            # Data exists within a pre-determined temporal threshold
            if len(spatial_temporal_properties) > 0:
                spatial_temporal_properties['time'] = spatial_temporal_properties.apply(
                    lambda x: (transaction_date - x['Date']).total_seconds() / 86400, axis=1)

                '''KNN selection 选择属性相似的个体'''
                # Attribute similarity calculation
                attribute_data = spatial_temporal_properties.iloc[:, 5:].drop(['Price', 'distance', 'time'], axis=1)
                sub_result = attribute_data.sub(target_property[5:-1])

                # Attribute distance calculation (Euclidean distance or Cosine similarity)
                attribute_data['attribute_distance'] = sub_result.apply(lambda x: np.sqrt(np.sum(x ** 2)), axis=1)

                if len(attribute_data) > n_neighbors:
                    # Select the most similar properties
                    initial_selection_neighbors = attribute_data.nsmallest(n_neighbors, 'attribute_distance')
                    initial_selection_neighbors = spatial_temporal_properties.iloc[initial_selection_neighbors.index.tolist(), :]
                else:
                    initial_selection_neighbors = spatial_temporal_properties

                '''
                ADJUSTMENT 
                对于选择得到的个体表现进行预测
                '''

                '''GTWR Weight Creation 创建GTWR权重函数'''
                # Gaussian weight function
                if gtwr_weight == "gaussian":
                    # Spatial Weight Calculation
                    initial_selection_neighbors['spatial_weight'] = initial_selection_neighbors.apply(
                        lambda x: math.exp(-np.square(x['distance'] / spatial)), axis=1)

                    # Temporal Weight Calculation
                    initial_selection_neighbors['temporal_weight'] = initial_selection_neighbors.apply(
                        lambda x: math.exp(-np.square(x['time'] / temporal)), axis=1)

                    # GTWR Weight Calculation
                    initial_selection_neighbors['GTWR_weight'] = initial_selection_neighbors.apply(
                        lambda x: x['spatial_weight'] * x['temporal_weight'], axis=1)

                # Bi-Square Weight Function
                elif gtwr_weight == "bi-square":

                    # Spatial Weight Calculation
                    initial_selection_neighbors['spatial_weight'] = initial_selection_neighbors.apply(
                        lambda x: np.square(1 - math.exp(np.square(x['distance']) / np.square(spatial))), axis=1)

                    # Temporal Weight Calculation
                    initial_selection_neighbors['temporal_weight'] = initial_selection_neighbors.apply(
                        lambda x: np.square(1 - math.exp(np.square(x['time']) / np.square(temporal))), axis=1)

                    # GTWR Weight Calculation
                    initial_selection_neighbors['GTWR_weight'] = initial_selection_neighbors.apply(
                        lambda x: x['spatial_weight'] * x['temporal_weight'], axis=1)

                # Standardization of the GTWR weights
                # MinMax Scaler
                standardized_data = MinMaxScaler().fit_transform(initial_selection_neighbors['GTWR_weight'].values.reshape(-1, 1))
                initial_selection_neighbors['GTWR_weight_standard'] = standardized_data

                ''' 
                Value aggregation 整合选择得到的个体表现得到预测值
                '''
                weight = initial_selection_neighbors.loc[:, 'GTWR_weight_standard'].reset_index(drop=True)
                price = spatial_temporal_properties.loc[initial_selection_neighbors.index.tolist(), 'Price'].reset_index(drop=True)
                predicted_price = weight.dot(price)

            # 空间包含但是时间不包含
            else:
                # Attribute similarity calculation
                attribute_data = spatial_properties.iloc[:, 5:-1]
                sub_result = attribute_data.sub(target_property[5:-1])

                if len(attribute_data) > n_neighbors:
                    # Select the most similar properties
                    # Attribute distance calculation (Euclidean distance or Cosine similarity)
                    attribute_data['attribute_distance'] = sub_result.apply(lambda x: np.sqrt(np.sum(x ** 2)), axis=1)
                    initial_selection_neighbors = attribute_data.nsmallest(n_neighbors, 'attribute_distance')

                else:
                    initial_selection_neighbors = spatial_properties

                price = initial_selection_neighbors.loc[initial_selection_neighbors.index.tolist(), 'Price'].reset_index(drop=True)

                # KNN hyperparameter configuration
                # ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
                if knn_weight == 'uniform':
                    predicted_price = price.mean()

                # ‘distance’ : weight points by the inverse of their distance.
                # In this case, closer neighbors of a query point will have a greater
                # influence than neighbors which are further away.
                elif knn_weight == 'distance':
                    weights = attribute_data['attribute_distance'].apply(lambda x: 1 / x).reset_index(drop=True)
                    standardized_weights = weights.apply(lambda x: x / weights.sum()).reset_index(drop=True)
                    predicted_price = standardized_weights.dot(price)

    # 如果地理范围内不存在数据，采用属性KNN计算
    else:
        # increase spatial search range and return sufficient samples for attribute similarity check
        delta_distance = 100

        i = 0
        while i < 100:
            initial_selection = other_properties[(other_properties['x'] <= target_property['x'] + spatial + i * delta_distance) &
                                                    (other_properties['x'] >= target_property['x'] - spatial - i * delta_distance) &
                                                    (other_properties['y'] <= target_property['y'] + spatial + i * delta_distance) &
                                                    (other_properties['y'] >= target_property['y'] - spatial - i * delta_distance)]
            i += 1
            if len(initial_selection) < n_neighbors * 2:
                continue
            else:
                break

        # Attribute distance calculation (Euclidean distance or Cosine similarity)
        attribute_data = initial_selection.iloc[:, 5:-1]
        sub_result = attribute_data.sub(target_property[5:-1])

        attribute_data['attribute_distance'] = sub_result.apply(lambda x: np.sqrt(np.sum(x ** 2)), axis=1)
        initial_selection_neighbors = attribute_data.nsmallest(n_neighbors, 'attribute_distance')
        price = train.loc[initial_selection_neighbors.index.tolist(), 'Price'].reset_index(drop=True)

        # KNN hyperparameter configuration
        # ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
        if knn_weight == 'uniform':
            predicted_price = price.mean()

        # ‘distance’ : weight points by the inverse of their distance.
        # Closer neighbors of a query point will have a greater influence than neighbors which are further away.
        elif knn_weight == 'distance':
            weights = initial_selection_neighbors['attribute_distance'].apply(lambda x: 1 / x).reset_index(drop=True)
            standardized_weights = weights.apply(lambda x: x / weights.sum()).reset_index(drop=True)
            predicted_price = standardized_weights.dot(price)

    return predicted_price


# Derive predictions of the test data set
def gtwr_knn_prediction(train, test, num, spatial, temporal, gtwr_weight, n_neighbors, knn_weight):
    test = test.reset_index(drop=True)

    train['Date'] = pd.to_datetime(train['Date'], format='%d/%m/%Y')
    test['Date'] = pd.to_datetime(test['Date'], format='%d/%m/%Y')

    """
    SELECTION 选择
    """

    '''Initial selection 选择时空相近的个体'''
    # target property
    target_property = test.iloc[num, :]

    # Other properties
    other_properties = train

    # Spatial_proximity
    rectangle_properties = other_properties[(other_properties['x'] <= target_property['x'] + spatial) &
                                            (other_properties['x'] >= target_property['x'] - spatial) &
                                            (other_properties['y'] <= target_property['y'] + spatial) &
                                            (other_properties['y'] >= target_property['y'] - spatial)]

    if len(rectangle_properties) > 0:
        rectangle_properties['distance'] = rectangle_properties.apply(lambda x: geodesic((
            target_property['Latitude'],
            target_property['Longitude']),
            (x['Latitude'],
             x['Longitude'])).m,
            axis=1)

        # Select properties within the distance threshold
        spatial_properties = rectangle_properties[rectangle_properties['distance'] <= spatial]

        if len(spatial_properties) > 0:

            # Temporal gap: needs to be larger to cover more data
            delta = timedelta(days=temporal)

            transaction_date = target_property['Date']
            start_time = transaction_date - delta

            # Select the properties within the selected time period
            # If there is no record with specific period, then choose the data with spatial threshold

            spatial_temporal_properties = spatial_properties[(spatial_properties['Date'] >= start_time) &
                                                             (spatial_properties['Date'] <= transaction_date)
                                                             ].reset_index(drop=True)

            # Data exists within a pre-determined temporal threshold
            if len(spatial_temporal_properties) > 0:
                spatial_temporal_properties['time'] = spatial_temporal_properties.apply(
                    lambda x: (transaction_date - x['Date']).total_seconds() / 86400, axis=1)

                '''KNN selection 选择属性相似的个体'''
                # Attribute similarity calculation
                attribute_data = spatial_temporal_properties.iloc[:, 5:].drop(['Price', 'distance', 'time'], axis=1)
                sub_result = attribute_data.sub(target_property[5:-1])

                # Attribute distance calculation (Euclidean distance or Cosine similarity)
                attribute_data['attribute_distance'] = sub_result.apply(lambda x: np.sqrt(np.sum(x ** 2)), axis=1)

                if len(attribute_data) > n_neighbors:
                    # Select the most similar properties
                    initial_selection_neighbors = attribute_data.nsmallest(n_neighbors, 'attribute_distance')
                    initial_selection_neighbors = spatial_temporal_properties.iloc[
                                                  initial_selection_neighbors.index.tolist(), :]
                else:
                    initial_selection_neighbors = spatial_temporal_properties

                '''
                ADJUSTMENT 
                对于选择得到的个体表现进行预测
                '''

                '''GTWR Weight Creation 创建GTWR权重函数'''
                # Gaussian weight function
                if gtwr_weight == "gaussian":
                    # Spatial Weight Calculation
                    initial_selection_neighbors['spatial_weight'] = initial_selection_neighbors.apply(
                        lambda x: math.exp(-np.square(x['distance'] / spatial)), axis=1)

                    # Temporal Weight Calculation
                    initial_selection_neighbors['temporal_weight'] = initial_selection_neighbors.apply(
                        lambda x: math.exp(-np.square(x['time'] / temporal)), axis=1)

                    # GTWR Weight Calculation
                    initial_selection_neighbors['GTWR_weight'] = initial_selection_neighbors.apply(
                        lambda x: x['spatial_weight'] * x['temporal_weight'], axis=1)

                # Bi-Square Weight Function
                elif gtwr_weight == "bi-square":

                    # Spatial Weight Calculation
                    initial_selection_neighbors['spatial_weight'] = initial_selection_neighbors.apply(
                        lambda x: np.square(1 - math.exp(np.square(x['distance']) / np.square(spatial))), axis=1)

                    # Temporal Weight Calculation
                    initial_selection_neighbors['temporal_weight'] = initial_selection_neighbors.apply(
                        lambda x: np.square(1 - math.exp(np.square(x['time']) / np.square(temporal))), axis=1)

                    # GTWR Weight Calculation
                    initial_selection_neighbors['GTWR_weight'] = initial_selection_neighbors.apply(
                        lambda x: x['spatial_weight'] * x['temporal_weight'], axis=1)

                # Standardization of the GTWR weights
                # MinMax Scaler
                standardized_data = MinMaxScaler().fit_transform(
                    initial_selection_neighbors['GTWR_weight'].values.reshape(-1, 1))
                initial_selection_neighbors['GTWR_weight_standard'] = standardized_data

                ''' 
                Value aggregation 整合选择得到的个体表现得到预测值
                '''
                weight = initial_selection_neighbors.loc[:, 'GTWR_weight_standard'].reset_index(drop=True)
                price = spatial_temporal_properties.loc[
                    initial_selection_neighbors.index.tolist(), 'Price'].reset_index(drop=True)
                predicted_price = weight.dot(price)

            # 空间包含但是时间不包含
            else:
                # Attribute similarity calculation
                attribute_data = spatial_properties.iloc[:, 5:-1]
                sub_result = attribute_data.sub(target_property[5:-1])

                if len(attribute_data) > n_neighbors:
                    # Select the most similar properties
                    # Attribute distance calculation (Euclidean distance or Cosine similarity)
                    attribute_data['attribute_distance'] = sub_result.apply(lambda x: np.sqrt(np.sum(x ** 2)), axis=1)
                    initial_selection_neighbors = attribute_data.nsmallest(n_neighbors, 'attribute_distance')

                else:
                    initial_selection_neighbors = spatial_properties

                price = initial_selection_neighbors.loc[
                    initial_selection_neighbors.index.tolist(), 'Price'].reset_index(drop=True)

                # KNN hyperparameter configuration
                # ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
                if knn_weight == 'uniform':
                    predicted_price = price.mean()

                # ‘distance’ : weight points by the inverse of their distance.
                # In this case, closer neighbors of a query point will have a greater
                # influence than neighbors which are further away.
                elif knn_weight == 'distance':
                    weights = attribute_data['attribute_distance'].apply(lambda x: 1 / x).reset_index(drop=True)
                    standardized_weights = weights.apply(lambda x: x / weights.sum()).reset_index(drop=True)
                    predicted_price = standardized_weights.dot(price)

    # 如果地理范围内不存在数据，采用属性KNN计算
    else:
        # increase spatial search range and return sufficient samples for attribute similarity check
        delta_distance = 100

        i = 0
        while i < 100:
            initial_selection = other_properties[
                (other_properties['x'] <= target_property['x'] + spatial + i * delta_distance) &
                (other_properties['x'] >= target_property['x'] - spatial - i * delta_distance) &
                (other_properties['y'] <= target_property['y'] + spatial + i * delta_distance) &
                (other_properties['y'] >= target_property['y'] - spatial - i * delta_distance)]
            i += 1
            if len(initial_selection) < n_neighbors * 2:
                continue
            else:
                break

        # Attribute distance calculation (Euclidean distance or Cosine similarity)
        attribute_data = initial_selection.iloc[:, 5:-1]
        sub_result = attribute_data.sub(target_property[5:-1])

        attribute_data['attribute_distance'] = sub_result.apply(lambda x: np.sqrt(np.sum(x ** 2)), axis=1)
        initial_selection_neighbors = attribute_data.nsmallest(n_neighbors, 'attribute_distance')
        price = train.loc[initial_selection_neighbors.index.tolist(), 'Price'].reset_index(drop=True)

        # KNN hyperparameter configuration
        # ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
        if knn_weight == 'uniform':
            predicted_price = price.mean()

        # ‘distance’ : weight points by the inverse of their distance.
        # Closer neighbors of a query point will have a greater influence than neighbors which are further away.
        elif knn_weight == 'distance':
            weights = initial_selection_neighbors['attribute_distance'].apply(lambda x: 1 / x).reset_index(drop=True)
            standardized_weights = weights.apply(lambda x: x / weights.sum()).reset_index(drop=True)
            predicted_price = standardized_weights.dot(price)

    return predicted_price


def parallel_training(train_data, spatial, temporal, gtwr_weight, n_neighbors, knn_weight, n_jobs):
    cpu_num = range(1, multiprocessing.cpu_count() + 1)
    processor_num = cpu_num[n_jobs - 1] if n_jobs > 0 else cpu_num[n_jobs]

    training_result = []
    # Grid search
    for i in spatial:
        for j in temporal:
            for k in n_neighbors:
                '''
                Parallel Computing 并行计算进行训练
                '''
                data_count = len(train_data)
                pbar = tqdm(total=data_count,
                            file=sys.stdout,
                            desc='GTWR_KNN_Training',
                            colour='white')

                update = lambda *args: pbar.update()

                # VERY IMPORTANT: check how many cores in your PC
                pool = multiprocessing.Pool(processes=processor_num)

                # 定义一个列表来存储每次循环的结果
                results = []

                # 并行运行for循环
                for num in range(data_count):
                    # 将任务提交给进程池
                    result = pool.apply_async(gtwr_knn_training,
                                              args=(train_data,
                                                    num,
                                                    i,
                                                    j,
                                                    gtwr_weight,
                                                    k,
                                                    knn_weight),
                                              callback=update)
                    results.append(result)

                # 等待所有进程完成
                pool.close()
                pool.join()
                # print('Time: {} seconds'.format(time.time()-start))

                pred_results = []
                # 打印每次循环的结果
                for result in results:
                        pred_results.append(result.get())

                pred_results = pd.DataFrame(pred_results)

                pred_results.to_csv('./training_results/spatial_{}_temporal_{}_n_{}_prediction_results.csv'.
                                    format(i,
                                           j,
                                           k),
                                    index=False,
                                    header=False)

                performance_result = model_performance(train_data.iloc[:, -1], pred_results)
                performance_result['spatial_bandwidth'] = i
                performance_result['temporal_bandwidth'] = j
                performance_result['n_neighbors'] = k

                training_result.append(performance_result)

    best_parameters = min(training_result, key=lambda x: x['MAE'])
    print('Best Parameters: ', best_parameters)
    return best_parameters

'''
Parallel Computing 并行计算进行结果预测
'''


def parallel_testing(train_data, test_data, spatial, temporal, gtwr_weight, n_neighbors, knn_weight, n_jobs):

    cpu_num = range(1, multiprocessing.cpu_count() + 1)
    processor_num = cpu_num[n_jobs - 1] if n_jobs > 0 else cpu_num[n_jobs]

    # Testing
    data_count = len(test_data)
    pbar = tqdm(total=data_count)
    pbar.set_description('GTWR_KNN_Testing')
    update = lambda *args: pbar.update()

    # VERY IMPORTANT: check how many cores in your PC
    pool = multiprocessing.Pool(processes=processor_num)

    # 定义一个列表来存储每次循环的结果
    results = []

    # 并行运行for循环
    for num in range(data_count):
        # 将任务提交给进程池
        result = pool.apply_async(gtwr_knn_prediction,
                                  args=(train_data,
                                        test_data,
                                        num,
                                        spatial,
                                        temporal,
                                        gtwr_weight,
                                        n_neighbors,
                                        knn_weight),
                                  callback=update)
        results.append(result)

    # 等待所有进程完成
    pool.close()
    pool.join()

    pred_results = []
    # 打印每次循环的结果
    for result in results:
        pred_results.append(result.get())

    pred_results = pd.DataFrame(pred_results)
    testing_results = model_performance(test_data.iloc[:, -1], pred_results)
    print(testing_results)

    return testing_results


if __name__ == "__main__":
    walk_threshold = 400
    data_features = pd.read_csv('data_features.csv', encoding='unicode_escape')
    train_data = data_split(data=data_features, threshold=walk_threshold)[0]
    test_data = data_split(data=data_features, threshold=walk_threshold)[1]

    # candidate for grid search (500, 1000, 1500, 2000)
    spatial_bandwidth_candidates = [500, 1000, 1500, 2000]

    # candidate for grid search (30, 45, 60, 75, 90)
    temporal_bandwidth_candidates = [30, 45, 60, 75, 90]

    # candidate for grid search (5, 10, 15)
    n_neighbors = [5, 7, 9, 11, 13, 15]

    # get the best parameters from grid search strategy
    best_parameters = parallel_training(
                      train_data=train_data,
                      spatial=spatial_bandwidth_candidates,
                      temporal=temporal_bandwidth_candidates,
                      gtwr_weight='bi-square',
                      n_neighbors=n_neighbors,
                      knn_weight='uniform',
                      n_jobs=-1)

    # Derive predictions on testing set
    predictions = parallel_testing(train_data=train_data,
                                   test_data=test_data,
                                   spatial=best_parameters['spatial_bandwidth'],
                                   temporal=best_parameters['temporal_bandwidth'],
                                   gtwr_weight='bi-square',
                                   n_neighbors=best_parameters['n_neighbors'],
                                   knn_weight='uniform',
                                   n_jobs=-1)



