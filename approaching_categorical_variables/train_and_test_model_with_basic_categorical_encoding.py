from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import decomposition, ensemble, linear_model, metrics, preprocessing
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import utils
import xgboost as xgb


def runner(path: str, is_cat: bool, include_num_cols: bool):
    df = _load_dataset(path)
    lr = linear_model.LogisticRegression()
    rf = ensemble.RandomForestClassifier(n_jobs=1)
    xg = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=7,
        n_estimators=200
    )
    data_df, cat_features, num_features = _clean_dataset(df, is_cat, include_num_cols)

    # for fold in range(5):
    #     _run_fold_ohe_lr(data_df, fold, lr, cat_features, num_features)
    
    # for fold in range(5):
    #     _run_fold_lbl_enc_rf(data_df, fold, cat_features, num_features)
    
    # for fold in range(5):
    #     _run_fold_ohe_svd_rf(data_df, fold, cat_features, num_features)

    # for fold in range(5):
    #     _run_fold_lbl_xg(data_df, fold, xg, cat_features, num_features)

    for fold in range(5):
        _run_fold_entity_nn(data_df, fold, cat_features, num_features)


def _load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _clean_dataset(
    data_df: pd.DataFrame,
    is_cat: bool,
    include_num_feats: bool
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    num_features = []
    if not is_cat:
        # Removing Numeric COlumns for simplicity
        num_features = [
            "fnlwgt",
            "age",
            "capital.gain",
            "capital.loss",
            "hours.per.week"
        ]
        target_mapping = {
            "<=50K": 0,
            ">50K": 1
        }
        data_df = data_df.rename(columns={'income': 'target'})
        data_df.loc[:, 'target'] = data_df['target'].map(target_mapping)

    if not include_num_feats and num_features:
        data_df = data_df.drop(num_features, axis = 1)
        
    
    cat_features = [
        col for col in data_df.columns
        if col not in ('id', 'target', 'kfold')
        and col not in num_features 
    ]
    for col in cat_features:
        data_df.loc[:, col] = data_df[col].astype(str).fillna('None')
    
    return data_df, cat_features, num_features


def _run_fold_ohe_lr(
    data_df: pd.DataFrame,
    fold: int, 
    lr_model: linear_model.LogisticRegression,
    cat_features: List[str],
    num_features: List[str] = []
) -> None:
    ohe = preprocessing.OneHotEncoder(sparse=False)
    ohe.fit(data_df[cat_features])

    training_df = data_df[data_df['kfold'] != fold].reset_index(drop=True)
    test_df = data_df[data_df['kfold'] == fold].reset_index(drop=True)

    x_train_cat = ohe.transform(training_df[cat_features])
    x_test_cat = ohe.transform(test_df[cat_features])

    x_train_num = training_df[num_features].values if num_features else []
    x_test_num = test_df[num_features].values if num_features else []

    if num_features:
        x_train = np.hstack((x_train_cat, x_train_num))
        x_test = np.hstack((x_test_cat, x_test_num))
    else:
        x_train = x_train_cat
        x_test = x_test_cat

    lr_model.fit(x_train, training_df['target'].values)
    test_preds = lr_model.predict_proba(x_test)[:, 1]
    auc_score = metrics.roc_auc_score(test_df['target'].values, test_preds)

    print(f'Fold: {fold}, Model: One-Hot Encoding + Logistic Regression, AUC Score: {auc_score}')


def _run_fold_lbl_enc_rf(
    data_df: pd.DataFrame,
    fold: int, 
    rf_model: ensemble.RandomForestClassifier,
    cat_features: List[str],
    num_features: List[str]
) -> None:
    features = cat_features + num_features
    for col in cat_features:
        lbl_enc = preprocessing.LabelEncoder()
        lbl_enc.fit(data_df[col])
        data_df.loc[:, col] = lbl_enc.transform(data_df[col])

    training_df = data_df[data_df['kfold'] != fold].reset_index(drop=True)
    test_df = data_df[data_df['kfold'] == fold].reset_index(drop=True)


    rf_model.fit(training_df[features].values, training_df['target'].values)
    test_preds = rf_model.predict_proba(test_df[features].values)[:, 1]
    auc_score = metrics.roc_auc_score(test_df['target'].values, test_preds)

    print(f'Fold: {fold}, Model: Label Encoding + Random Forest, AUC Score: {auc_score}')


def _run_fold_ohe_svd_rf(
    data_df: pd.DataFrame,
    fold: int, 
    rf_model: ensemble.RandomForestClassifier,
    cat_features: List[str],
    num_features: List[str]
) -> None:
    features = cat_features + num_features
    ohe = preprocessing.OneHotEncoder(sparse=False)
    ohe.fit(data_df[cat_features])

    training_df = data_df[data_df['kfold'] != fold].reset_index(drop=True)
    test_df = data_df[data_df['kfold'] == fold].reset_index(drop=True)

    x_train_cat = ohe.transform(training_df[cat_features])
    x_test_cat = ohe.transform(test_df[cat_features])

    x_train_num = training_df[num_features].values if num_features else []
    x_test_num = test_df[num_features].values if num_features else []

    if num_features:
        x_train = np.hstack((x_train_cat, x_train_num))
        x_test = np.hstack((x_test_cat, x_test_num))
    else:
        x_train = x_train_cat
        x_test = x_test_cat


    svd = decomposition.TruncatedSVD(n_components=120)

    full_sparse = sparse.vstack((x_train, x_test))
    svd.fit(full_sparse)

    x_train = svd.transform(x_train)
    x_test = svd.transform(x_test)

    rf_model.fit(training_df[features].values, training_df['target'].values)
    test_preds = rf_model.predict_proba(test_df[features].values)[:, 1]
    auc_score = metrics.roc_auc_score(test_df['target'].values, test_preds)

    print(f'Fold: {fold}, Model: One-Hot Encoding + SVD + Random Forest, AUC Score: {auc_score}')


def _run_fold_lbl_xg(
    data_df: pd.DataFrame,
    fold: int, 
    xg_boost_model: xgb.XGBClassifier,
    cat_features: List[str],
    num_features: List[str]
) -> None:
    features = cat_features + num_features
    for col in cat_features:
        lbl_enc = preprocessing.LabelEncoder()
        lbl_enc.fit(data_df[col])
        data_df.loc[:, col] = lbl_enc.transform(data_df[col])

    training_df = data_df[data_df['kfold'] != fold].reset_index(drop=True)
    test_df = data_df[data_df['kfold'] == fold].reset_index(drop=True)


    xg_boost_model.fit(training_df[features].values, training_df['target'].values)
    test_preds = xg_boost_model.predict_proba(test_df[features].values)[:, 1]
    auc_score = metrics.roc_auc_score(test_df['target'].values, test_preds)

    print(f'Fold: {fold}, Model: Label Encoding + XGBoost, AUC Score: {auc_score}')


def _run_fold_entity_nn(
    data_df: pd.DataFrame,
    fold: int, 
    cat_features: List[str],
    num_features: List[str]
) -> None:
    features = cat_features + num_features
    for col in cat_features:
        lbl_enc = preprocessing.LabelEncoder()
        lbl_enc.fit(data_df[col])
        data_df.loc[:, col] = lbl_enc.transform(data_df[col])

    training_df = data_df[data_df['kfold'] != fold].reset_index(drop=True)
    test_df = data_df[data_df['kfold'] == fold].reset_index(drop=True)

    model = _get_entity_model(data_df, cat_features, num_features)

    # Creates N 1D arrays.
    x_train = [
        training_df[features].values[:, k] for k in range(len(features))
    ]
    x_test = [
        test_df[features].values[:, k] for k in range(len(features))
    ]

    y_train = training_df['target'].values
    y_test = test_df['target'].values

    # convert target columns to categories
    # this is just binarization
    y_train_enc = utils.to_categorical(y_train)
    y_valid_enc = utils.to_categorical(y_test)

    model.fit(
        x_train,
        y_train_enc,
        validation_data=(x_test, y_valid_enc),
        verbose=1,
        batch_size=1024,
        epochs=30
    )

    test_preds = model.predict(x_test)[:, 1]
    auc_score = metrics.roc_auc_score(test_df['target'].values, test_preds)

    print(f'Fold: {fold}, Model: Entity Encoding + NN, AUC Score: {auc_score}')
    K.clear_session()


def _get_entity_model(
    data_df: pd.DataFrame,
    cat_features: List[str],
    num_features: List[str]
) -> Model:
    inputs = []
    outputs = []

    for feat in cat_features:
        num_unique_cat = data_df[feat].nunique()
        
        # Setting the max value as 50.
        embedding_dimensions = min(
            num_unique_cat // 2, 50
        )

        # Creating an Input Layer. It only accepts one value
        # (The category) so it needs to be of shape (1, ).

        # Note that the category would be mapped to a number before
        # sending it to a NN.

        input_layer = layers.Input(shape=(1,))

        # Embedding Layer requires the number of unique categories, the embedding dimensions
        # and a name is given to distinguish it from other embedding layers.
        embedding_layer = layers.Embedding(num_unique_cat + 1, embedding_dimensions, name=feat)(input_layer)

        # Here, 30% of the embedding features are randomly set to zero. This is to prevent Overfitting.
        droput_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

        reshape_layer = layers.Reshape((embedding_dimensions, ))(droput_layer)

        inputs.append(input_layer)
        outputs.append(reshape_layer)
    
    for feat in num_features:
        input_layer = layers.Input(shape=(1, ))
        batch_norm_layer = layers.BatchNormalization()(input_layer)

        inputs.append(input_layer)
        outputs.append(batch_norm_layer)
    

    # Concatentating the Embeddings
    x = layers.Concatenate()(outputs)

    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(300, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(300, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    y = layers.Dense(2, activation='softmax')(x)

    model = Model(inputs, y)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


if __name__ == '__main__':
    CAT_PATH = 'approaching_categorical_variables/inputs/cat_train_folds.csv'
    ADULT_PATH = 'approaching_categorical_variables/inputs/adult_folds.csv'
    # runner(CAT_PATH, True, False)
    runner(ADULT_PATH, False, True)