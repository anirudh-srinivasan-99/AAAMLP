from sklearn.datasets import fetch_california_housing
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    data = fetch_california_housing()
    x = data['data']
    features = data['feature_names']
    y = data['target']

    model = LinearRegression()
    rfe = RFE(estimator=model, n_features_to_select=3)

    rfe.fit(x, y)
    x_transformed = rfe.transform(x)
    print(f'X before feature selection {x.shape}')
    print(f'X after feature selection: {x_transformed.shape}')