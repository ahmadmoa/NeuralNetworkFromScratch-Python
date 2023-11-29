from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import  pandas as pd
def preprocess_dataSet(drybean_Data):
    # Scaling Numerical Data
    scaler = MinMaxScaler()
    columns_toBe_Scalled = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']
    drybean_Data[columns_toBe_Scalled] = scaler.fit_transform(drybean_Data.loc[:, columns_toBe_Scalled].to_numpy())
    drybean_Data[columns_toBe_Scalled] = pd.DataFrame(drybean_Data[columns_toBe_Scalled],columns=columns_toBe_Scalled)
    # Encoding Categorical Data
    label_encoder = LabelEncoder()
    drybean_Data['Class'] = label_encoder.fit_transform(drybean_Data['Class'])

    # Separating each class
    Class1_Data = drybean_Data[0:50].copy()
    Class2_Data = drybean_Data[50:100].copy()
    Class3_Data = drybean_Data[100:150].copy()

    numerical_columns_Names =  ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']
    for colum in numerical_columns_Names:
        mean_value = Class1_Data[colum].mean()
        Class1_Data[colum].fillna(value=mean_value, inplace=True)

        mean_value = Class2_Data[colum].mean()
        Class2_Data[colum].fillna(value=mean_value, inplace=True)

        mean_value = Class3_Data[colum].mean()
        Class3_Data[colum].fillna(value=mean_value, inplace=True)

    return Class1_Data, Class2_Data, Class3_Data


# def preprocess(classes, features):
#     # Read the dataset
#     data = pd.read_csv('data/raw/Dry_Bean_Dataset.csv')
#     # Filter the dataset
#     data = data[data['Class'].isin(classes)]
#     # Perform linear interpolation for missing values
#     data[features] = data[features].interpolate(method='linear', axis=0, limit_direction='both')
#     # Manually perform Min-Max scaling
#     for column in features:
#         min_val = data[column].min()
#         max_val = data[column].max()
#         data[column] = (data[column] - min_val) / (max_val -  min_val)
#
#     # Shuffle the data
#     # data = data.sample (frac=1, random_state=0).reset_index(drop=True) data = shuffle (data, random_state=0)
#     X = data[features].values
#     Y = np.where(data['class']==classes[0], -1, 1)
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
#     return x_train, x_test, y_train, y_test