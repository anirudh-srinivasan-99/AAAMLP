WINE_DATASET_PATH = 'cross_validation/input/winequality-red.csv'
TARGET_VALUE_MAPPING = {
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5
}
FEATURE_COLUMNS = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]
TARGET_COLUMN = 'quality'