from src.processing import DatasetProcessor
from src.train_eval import ModelTrainer


CSV_FILE = "dataset/churn_data.csv"
MODEL_TYPE = "XGBoost"

if __name__ == "__main__":
    processed_df = DatasetProcessor(CSV_FILE).process()
    trainer = ModelTrainer(processed_df, MODEL_TYPE)
    model = trainer.train()
    trainer.evaluate()
    trainer.lime_explain()
