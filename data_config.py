class ConfigData:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.csv_path = f"{data_dir}/BTC_1sec_with_sentiment_risk_train.csv"
        self.predict_ary_path = f"{data_dir}/BTC_1sec_predict.npy"