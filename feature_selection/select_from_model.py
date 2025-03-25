from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    data = load_diabetes()
    x = data['data']
    features = data['feature_names']
    y = data['target']

    model = LinearRegression()
    sfm = SelectFromModel(estimator=model)

    sfm.fit(x, y)
    x_transformed = sfm.transform(x)
    print(f'X before feature selection {x.shape}')
    print(f'X after feature selection: {x_transformed.shape}')