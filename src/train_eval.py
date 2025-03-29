import json
from pathlib import Path
from typing import Literal
from dataclasses import dataclass

import joblib
import lime
import pickle
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


@dataclass
class XYSplits:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame
    x_train_scaled: pd.DataFrame = None
    x_test_scaled: pd.DataFrame = None
    columns: pd.Index = None


class ModelTrainer:
    name_model_map = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    }

    def __init__(
        self,
        df: pd.DataFrame,
        model_type: Literal["XGBoost", "RandomForest"] = "XGBoost",
    ):
        """
        Initialize the ModelTrainer with the dataset, model type, and output directory.

        Args:
            df (pd.DataFrame): The input DataFrame containing the dataset.
            model_type (Literal["XGBoost", "RandomForest"], optional): The model type to use. 
                Defaults to "XGBoost". Must be one of the keys in `name_model_map`.
            
        Raises:
            ValueError: If `model_type` is not recognized.

        Attributes:
            name (str): The name of the model type.
            df (pd.DataFrame): The input DataFrame.
            label_encoders (dict or None): The label encoders for categorical columns.
            splits (XYSplits or None): The training and testing data splits.
            model: The initialized machine learning model.
            results_dir (Path): The path to the results directory.
            models_dir (Path): The path to the models directory.
        """

        if model_type not in self.name_model_map:
            raise ValueError("Unknown model")
        self.name = model_type
        self.df = df
        self.label_encoders = None
        self.splits = None
        self.model = self.name_model_map[model_type]

        self.results_dir = Path("output")
        self.models_dir = Path("models")

        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.models_dir.mkdir(exist_ok=True, parents=True)

    def preprocess(self):
        """
        Preprocess the data by performing the following steps:
            1. Label encoding: encode categorical columns using LabelEncoder.
            2. NaN replacement: replace infinite values with NaN.
            3. Mean imputation: replace NaN values with the mean of the column using SimpleImputer.

        The preprocessed DataFrame is stored in `self.df`, and the label encoders are stored in `self.label_encoders`.
        """
        categoricalCols = self.df.select_dtypes(include=["object"]).columns
        labelEncoders = {}

        for col in categoricalCols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            labelEncoders[col] = le

        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        imputer = SimpleImputer(strategy="mean")

        self.df = pd.DataFrame(imputer.fit_transform(self.df), columns=self.df.columns)
        self.label_encoders = labelEncoders
        
        with open(self.models_dir / "label_encoders.pkl", "wb") as f:
            pickle.dump(self.label_encoders, f)

    def split(self):
        """
        Split the preprocessed data into training and testing sets, then scale the data
        using StandardScaler. The scaled data is stored in `self.splits`.

        The scaled training data is stored in `self.splits.x_train_scaled`, and the scaled
        testing data is stored in `self.splits.x_test_scaled`. The original columns are stored
        in `self.splits.columns`. The scaler is saved to the `models_dir` directory.

        The splits are stored in `self.splits` as an XYSplits object.
        """
        X = self.df.drop(columns=["churn"])
        y = self.df["churn"]
        self.splits = XYSplits(*train_test_split(X, y, test_size=0.2, random_state=42))

        scaler = StandardScaler()
        self.splits.x_train_scaled = scaler.fit_transform(self.splits.x_train)
        self.splits.x_test_scaled = scaler.transform(self.splits.x_test)
        self.splits.columns = X.columns

        joblib.dump(scaler, self.models_dir / "scaler.pkl")

    def _train(self):
        """
        Train the machine learning model using the training data.

        This method fits the model to the training features and labels stored
        in `self.splits.x_train` and `self.splits.y_train`.
        """
        self.model.fit(self.splits.x_train, self.splits.y_train)

    def evaluate(self):
        """
        Evaluate the performance of the trained model using the test dataset.

        This method predicts the labels for the test dataset and calculates evaluation metrics
        including accuracy, F1 score, and a detailed classification report. The confusion matrix
        is visualized and saved as a heatmap. The evaluation metrics and the trained model are 
        saved to the output directory.

        Outputs:
            - Prints the model name, accuracy, and F1 score.
            - Saves the trained model as a pickle file in the models directory.
            - Saves the classification report as a JSON file in the results directory.
            - Saves the confusion matrix as an image file in the results directory.
        """
        y_pred = self.model.predict(self.splits.x_test)

        accuracy = accuracy_score(self.splits.y_test, y_pred)
        f1score = f1_score(self.splits.y_test, y_pred)
        report = classification_report(self.splits.y_test, y_pred, output_dict=True)
        report["f1_score"] = f1score

        print(f"\nModel: {self.name}")
        print("Accuracy:", accuracy)
        print("F1 Score:", f1score)
        print(classification_report(self.splits.y_test, y_pred))

        model_filename = self.name.lower()
        joblib.dump(self.model, self.models_dir / f"{model_filename}.pkl")

        report_path = self.results_dir / f"{model_filename}_metrics.json"
        with open(report_path, "w") as json_file:
            json.dump(report, json_file, indent=4)

        cm = confusion_matrix(self.splits.y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
        )
        plt.title(f"Confusion Matrix - {self.name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        fig_path = self.results_dir / f"{model_filename}_confusion_matrix.png"
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)
        self.model.save_model(self.models_dir / f"{self.name.lower()}_model.json")
        plt.close()


    def lime_explain(self):
        """
        Generate a LIME explanation for a randomly sampled instance from the test set.

        This method generates a LIME explanation for a randomly sampled instance from the test set.
        The explanation is visualized as a bar chart and saved as an image file in the results
        directory.

        Outputs:
            - Saves the explanation as an image file in the results directory.
        """
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(self.splits.x_train),
            feature_names=self.splits.columns.to_list(),
            class_names=["No Churn", "Churn"],
            discretize_continuous=True,
        )

        instance = np.array(self.splits.x_test.sample().iloc[0]).reshape(1, -1)

        exp = explainer.explain_instance(instance.flatten(), self.model.predict_proba)
        exp.as_pyplot_figure()
        plt.savefig(self.results_dir / "lime_explanation.png")
        plt.show()

    def train(self):
        """
        Execute the full training pipeline for the machine learning model.

        This method performs the following steps in sequence:
            1. Preprocess the input data.
            2. Split the data into training and testing sets.
            3. Train the model using the training data.

        Returns:
            The trained machine learning model.
        """
        self.preprocess()
        self.split()
        self._train()
        return self.model
