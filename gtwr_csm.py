"""
Selection and Adjustment

"""

import pandas as pd

'''
Data Processing
'''

# Import the data
data = pd.read_csv('final.csv')


def data_processing(data):
    # Delete unwanted column
    unwanted_columns = ['Source.Name', 'District_html', 'Estate_html', 'Unit Address', 'Contract', 'Cat_ID', 'Arearaw',
                        'Deal_ID']
    df = data.drop(unwanted_columns, axis=1)

    # Transaction date transformation
    df['Date'] = pd.to_datetime(df['Date'])

    # Outliers, missing values, inconsistent values

    # Outliers in price
    sorted_price = data['Price'].sort_values()  # sort the values as ascending order

    diff = sorted_price.diff()  # calculate the difference

    gap_point = sorted_price.loc[diff.idxmax()]  # find the data gap point

    df['Price'] = df['Price'].apply(lambda x: x / 1e6 if x >= gap_point else x)  # HK$ to Million HK$


if __name__ == "__main__":
    data_processing(data)
