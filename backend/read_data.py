import pandas as pd

class ReadData:
    def __init__(self, file):
        self.file = file

    def load_data(self):
        df = pd.read_csv(self.file)
        print("✅ Data Loaded Successfully")
        return df
