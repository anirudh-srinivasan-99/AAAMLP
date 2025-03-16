from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import manifold, datasets


def visualize_mnist_using_tsne():
    pixel_values, targets = _load_dataset()
    _plot_single_image(pixel_values[0])
    transformed_df = _transform_data(pixel_values[:3000], targets[:3000])
    _plot_transformed_data(transformed_df)


def _load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    dataset = datasets.fetch_openml(
        'mnist_784',
        version=1,
        return_X_y=True
    )
    pixel_values, targets = dataset
    targets = targets.astype(int)
    return pixel_values, targets


def _plot_single_image(flat_array: np.ndarray) -> None:
    single_image = flat_array.reshape(28, 28)
    plt.imshow(single_image, cmap='gray')


def _transform_data(pixel_values: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    tnse = manifold.TSNE(n_components=2, random_state=1)
    transformed_data = tnse.fit_transform(pixel_values)
    transformed_df = pd.DataFrame(
        np.column_stack(
            (transformed_data, targets)
        ),
        columns=['x', 'y', 'targets']
    )
    return transformed_df


def _plot_transformed_data(transformed_df: pd.DataFrame) -> None:
    grid = sns.FacetGrid(transformed_df, hue='targets', size = 8)
    grid.map(plt.scatter, "x", "y").add_legend()


if __name__ == '__main__':
    visualize_mnist_using_tsne()
    plt.show()
