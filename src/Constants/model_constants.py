from typing import List, Literal, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
)

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier, XGBRFClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from scikeras.wrappers import KerasClassifier

from src.Constants import common_constants


nn_arch_dict = {
    "neurons_layer_wise": [
        [16, 16, 16, 16, 6],
        [32, 32, 32, 32, 6],
        # [16, 32, 32, 16, 6],
        # [32, 16, 16, 32, 6],
        # [16, 32, 16, 32, 6],
        # [32, 16, 32, 16, 6],
        # [128, 64, 32, 16, 6],
    ],
    "activation_layer_wise": [
        ["elu", "elu", "elu", "elu", "softmax"],
        ["elu", "elu", "elu", "elu", "softmax"],
        # ["elu", "elu", "elu", "elu", "softmax"],
        # ["elu", "elu", "elu", "elu", "softmax"],
        # ["elu", "elu", "elu", "elu", "softmax"],
        # ["elu", "elu", "elu", "elu", "softmax"],
        # ["elu", "elu", "elu", "elu", "softmax"],
    ],
}


def create_model(
    index: int = 0,
    neurons_layer_wise: List[int] = [64, 32, 32, 1],
    activation_layer_wise: List[
        Literal["relu", "elu", "sigmoid", "softmax", "linear"]
    ] = ["relu", "relu", "relu", "sigmoid"],
    input_shape: Tuple[int, ...] = None,  # (10,),
    optimizer: Literal["adam", "sgd"] = "adam",
    loss: Literal[
        "binary_crossentropy", "categorical_crossentropy", "mean_squared_error"
    ] = "categorical_crossentropy",
    metrics: List[Literal["accuracy", "mse", "mae"]] = ["accuracy"],
):
    neurons_layer_wise = nn_arch_dict["neurons_layer_wise"][index]
    activation_layer_wise = nn_arch_dict["activation_layer_wise"][index]

    model = Sequential(
        [Input(shape=input_shape)]
        + [
            Dense(
                neurons_layer_wise[i],
                activation=activation_layer_wise[i],
            )
            for i in range(len(neurons_layer_wise))
        ]
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


nn_param_grid = {
    "index": range(len(nn_arch_dict["neurons_layer_wise"])),
    # "input_shape": [(10,)],  # [(x_train.shape[1],)],  #
    "optimizer": ["adam"],
    "loss": ["categorical_crossentropy"],
    "metrics": [["accuracy"]],
    "epochs": [100, 200],
    "batch_size": [10],
}

model_dict = {
    "LogisticRegression": {
        "Model": LogisticRegression(n_jobs=-1, random_state=common_constants.RANDOM),
        "Parameters": {
            # "penalty": ["l2"],
            # "max_iter": [100, 200]
        },
    },
    "KNeighborsClassifier": {
        "Model": KNeighborsClassifier(n_jobs=-1),
        "Parameters": {
            # "n_neighbors": [3, 5],
            # "leaf_size": [30, 50]
        },
    },
    "DecisionTreeClassifier": {
        "Model": DecisionTreeClassifier(random_state=common_constants.RANDOM),
        "Parameters": {
            # "max_depth": [None, 50],
            # "min_samples_split": [2, 3],
            # "min_samples_leaf": [1, 2],
        },
    },
    "AdaBoostClassifier": {
        "Model": AdaBoostClassifier(random_state=common_constants.RANDOM),
        "Parameters": {
            # "n_estimators": [50, 100],
            # "learning_rate": [1.0, 0.08],
        },
    },
    "GradientBoostingClassifier": {
        "Model": GradientBoostingClassifier(
            random_state=common_constants.RANDOM, criterion="friedman_mse"
        ),
        "Parameters": {
            # "loss": ["log_loss"],
            # "learning_rate": [0.1, 0.05],
            # "n_estimators": [100, 200],
            # "subsample": [1.0, 0.75],
            # "min_samples_split": [2, 3],
            # "min_samples_leaf": [1, 2],
            # "max_depth": [10, 50],
            # "n_iter_no_change": [5],
        },
    },
    "RandomForestClassifier": {
        "Model": RandomForestClassifier(
            n_jobs=-1, random_state=common_constants.RANDOM
        ),
        "Parameters": {
            # "n_estimators": [100, 200],
            # "max_depth": [None, 50],
            # "min_samples_split": [2, 3],
            # "min_samples_leaf": [1, 2],
            # "max_leaf_nodes": [None, 100],
        },
    },
    "HistGradientBoostingClassifier": {
        "Model": HistGradientBoostingClassifier(random_state=common_constants.RANDOM),
        "Parameters": {
            # "learning_rate": [0.1, 0.08],
            # "max_iter": [100, 200],
            # "max_leaf_nodes": [31, 50],
            # "max_depth": [None, 50],
            # "min_samples_leaf": [20, 50],
            # "l2_regularization": [0.0, 0.2],
        },
    },
    "ExtraTreesClassifier": {
        "Model": ExtraTreesClassifier(n_jobs=-1, random_state=common_constants.RANDOM),
        "Parameters": {
            # "n_estimators": [100, 200],
            # "max_depth": [None, 50],
            # "min_samples_split": [2, 3],
            # "min_samples_leaf": [1, 2],
            # "max_leaf_nodes": [None, 50],
        },
    },
    "LGBMClassifier": {
        "Model": LGBMClassifier(n_jobs=-1, random_state=common_constants.RANDOM),
        "Parameters": {
            # "boosting_type": ["gbdt", "rf"],
            # "num_leaves": [31, 50],
            # "max_depth": [-1, 50],
            # "learning_rate": [0.1, 0.08],
            # "n_estimators": [100, 200],
        },
    },
    "CatBoostClassifier": {
        "Model": CatBoostClassifier(random_seed=common_constants.RANDOM),
        "Parameters": {
            # "iterations": [200],
            # "learning_rate": [0.03, 0.08],
            # "depth": [6, 12],
            # "l2_leaf_reg": [1, 3],
            # "max_depth": [None, 50],
        },
    },
    "XGBClassifier": {
        "Model": XGBClassifier(
            n_jobs=-1, random_state=common_constants.RANDOM, verbosity=1
        ),
        "Parameters": {
            # "max_depth": [6, 12],
            # "max_leaves": [30, 50],
            # "learning_rate": [0.3, 0.1],
            # "n_estimators": [200],
            # "reg_lambda": [1, 3],
        },
    },
    "XGBRFClassifier": {
        "Model": XGBRFClassifier(
            n_jobs=-1, random_state=common_constants.RANDOM, verbosity=1
        ),
        "Parameters": {
            # "max_depth": [6, 12],
            # "max_leaves": [30, 50],
            # "learning_rate": [0.3, 0.1],
            # "n_estimators": [200],
            # "reg_lambda": [1, 3],
        },
    },
    "TFNeuralNetwork": {
        "Model": KerasClassifier(
            model=create_model,
            index=0,
            random_state=common_constants.RANDOM,
        ),
        "Parameters": nn_param_grid,
    },
}
