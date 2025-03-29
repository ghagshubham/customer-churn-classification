from pathlib import Path

import pandas as pd


INITIAL_DATE = "2024-03-01"


class DatasetProcessor:
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.df = pd.read_csv(self.csv_file)

        self.dataset_dir = Path("dataset")

        self.dataset_dir.mkdir(exist_ok=True, parents=True)


    def clean_dataset(self):
        """
        Clean the dataset by doing the following:
            1. Convert date and issuing date into datetime
            2. Sort the dataset by customer_id and date
            3. Fill missing transaction_amount with the mean of the customer's transaction_amounts
            4. Fill missing plan_type with the most frequent plan_type for each customer
        """
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df["issuing_date"] = pd.to_datetime(self.df["issuing_date"])

        self.df = self.df.sort_values(by=["customer_id", "date"])
        self.df["transaction_amount"] = self.df.groupby("customer_id")[
            "transaction_amount"
        ].transform(lambda x: x.fillna(x.mean()))

        most_frequent_plan = self.df.groupby("customer_id")["plan_type"].apply(
            lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
        )
        self.df["plan_type"] = self.df["plan_type"].fillna(
            self.df["customer_id"].map(most_frequent_plan)
        )


    def add_features(self):
        """
        Add features to the dataset.

        Features added:

        - `prev_transaction`: The customer's previous transaction amount
        - `transaction_trend`: The difference between the current and previous transaction amounts
        - `inactive_months`: The number of months the customer has been inactive
        - `first_transaction`: The date of the customer's first transaction
        - `last_transaction_date`: The date of the customer's last transaction
        - `days_since_last_txn`: The number of days since the customer's last transaction
        - `customer_tenure_days`: The number of days the customer has been with the company
        - `total_transactions`: The total number of transactions the customer has made
        - `avg_transaction_amount`: The average transaction amount for the customer
        - `std_transaction_amount`: The standard deviation of transaction amounts for the customer
        - `max_transaction`: The maximum transaction amount for the customer
        - `min_transaction`: The minimum transaction amount for the customer
        - `transaction_amount_3m_avg`: The average transaction amount for the customer over the past 3 months
        - `transaction_amount_6m_avg`: The average transaction amount for the customer over the past 6 months
        - `first_plan`: The customer's first plan type
        - `last_plan`: The customer's last plan type
        - `plan_switch_count`: The number of times the customer has switched plans
        - `premium_ratio`: The proportion of transactions that were premium
        - `standard_ratio`: The proportion of transactions that were standard
        - `basic_ratio`: The proportion of transactions that were basic
        - `is_downgraded`: Whether the customer has been downgraded
        - `is_upgraded`: Whether the customer has been upgraded
        - `cumulative_transaction`: The cumulative transaction amount for the customer
        - `transaction_level`: The customer's transaction level (based on cumulative transaction amount)
        - `transaction_gap`: The gap between the customer's last transaction date and the current date
        - `spending_variability`: The variability of the customer's spending
        - `loyalty_score`: A score based on customer tenure and cumulative transaction amount
        - `high_churn_risk`: Whether the customer has a high churn risk
        - `transaction_trend_score`: A score based on the customer's short vs long term trend
        """
        self.df["prev_transaction"] = self.df.groupby("customer_id")["transaction_amount"].shift(1)
        self.df["transaction_trend"] = self.df["transaction_amount"] - self.df["prev_transaction"]
        self.df["inactive_months"] = self.df.groupby("customer_id")["date"].diff().dt.days.fillna(0) / 30
        self.df["first_transaction"] = self.df.groupby("customer_id")["date"].transform("min")
        self.df["last_transaction_date"] = self.df.groupby("customer_id")["date"].transform("max")
        self.df["days_since_last_txn"] = (
            pd.to_datetime(INITIAL_DATE) - self.df["last_transaction_date"]
        ).dt.days

        self.df["customer_tenure_days"] = (self.df["date"] - self.df["issuing_date"]).dt.days
        self.df["total_transactions"] = self.df.groupby("customer_id")[
            "transaction_amount"
        ].transform("count")
        self.df["avg_transaction_amount"] = self.df.groupby("customer_id")[
            "transaction_amount"
        ].transform("mean")
        self.df["std_transaction_amount"] = (
            self.df.groupby("customer_id")["transaction_amount"].transform("std").fillna(0)
        )
        self.df["max_transaction"] = self.df.groupby("customer_id")["transaction_amount"].transform("max")
        self.df["min_transaction"] = self.df.groupby("customer_id")["transaction_amount"].transform("min")

        self.df["transaction_amount_3m_avg"] = (
            self.df.groupby("customer_id")["transaction_amount"]
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        self.df["transaction_amount_6m_avg"] = (
            self.df.groupby("customer_id")["transaction_amount"]
            .rolling(window=6, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        self.df["first_plan"] = self.df.groupby("customer_id")["plan_type"].transform("first")
        self.df["last_plan"] = self.df.groupby("customer_id")["plan_type"].transform("last")
        self.df["plan_switch_count"] = (
            self.df.groupby("customer_id")["plan_type"].transform("nunique") - 1
        )

        self.df["premium_ratio"] = self.df.groupby("customer_id")["plan_type"].transform(
            lambda x: (x == "Premium").sum() / len(x)
        )
        self.df["standard_ratio"] = self.df.groupby("customer_id")["plan_type"].transform(
            lambda x: (x == "Standard").sum() / len(x)
        )
        self.df["basic_ratio"] = self.df.groupby("customer_id")["plan_type"].transform(
            lambda x: (x == "Basic").sum() / len(x)
        )

        self.df["is_downgraded"] = (
            self.df.groupby("customer_id")["plan_type"]
            .apply(lambda x: ((x.shift(1) == "Premium") & (x != "Premium")).astype(int))
            .fillna(0).reset_index(drop=True)
        )
        self.df["is_upgraded"] = (
            self.df.groupby("customer_id")["plan_type"]
            .apply(lambda x: ((x.shift(1) != "Premium") & (x == "Premium")).astype(int))
            .fillna(0).reset_index(drop=True)
        )

        bins = list(range(0, 3300, 100)) + [float("inf")]
        labels = list(range(len(bins) - 1))

        self.df["cumulative_transaction"] = self.df.groupby("customer_id")[
            "transaction_amount"
        ].cumsum()
        self.df["transaction_level"] = pd.cut(
            self.df["cumulative_transaction"], bins=bins, labels=labels, right=True
        ).astype(int)

        self.df["transaction_gap"] = self.df["days_since_last_txn"] / self.df["customer_tenure_days"]
        self.df["spending_variability"] = self.df["std_transaction_amount"] / self.df["avg_transaction_amount"]
        self.df["loyalty_score"] = (
            self.df["customer_tenure_days"]
            * self.df["cumulative_transaction"]
            / (self.df["plan_switch_count"] + 1)
        )
        self.df["high_churn_risk"] = (self.df["inactive_months"] > 0.933333333333333).astype(
            int
        )  # Flag for customers inactive for >2 months
        self.df["transaction_trend_score"] = (
            self.df["transaction_amount_3m_avg"] - self.df["transaction_amount_6m_avg"]
        )  # Short vs Long term trend

        selected_features = [
            "customer_id",
            "transaction_amount",
            "transaction_gap",
            "spending_variability",
            "loyalty_score",
            "high_churn_risk",
            "transaction_trend_score",
            "plan_type",
            "is_downgraded",
            "is_upgraded",
            "total_transactions",
            "premium_ratio",
            "standard_ratio",
            "basic_ratio",
            "churn",
        ]
        self.df = self.df[selected_features]

    def process(self):
        """
        Process the dataset by cleaning and adding features.
        
        This function cleans the dataset by handling missing values, encoding categorical variables, and adding features.
        The cleaned dataset is saved to a csv file in the dataset directory.
        The function returns the cleaned dataset.
        """
        self.clean_dataset()
        self.add_features()
        self.df.to_csv(self.dataset_dir / "feature_dataset.csv", index=False)
        return self.df
    