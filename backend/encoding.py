from sklearn.preprocessing import LabelEncoder

class Encoding:
    def __init__(self, df):
        self.df = df

    def encode(self):
        le = LabelEncoder()
        for col in self.df.select_dtypes(include="object").columns:
            self.df[col] = le.fit_transform(self.df[col])
        print("✅ Encoding Completed")
        return self.df
