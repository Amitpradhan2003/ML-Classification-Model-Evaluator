class DataClean:
    def __init__(self, df):
        self.df = df

    def clean_data(self):
        self.df = self.df.drop_duplicates()
        print("✅ Duplicates Removed")
        return self.df
