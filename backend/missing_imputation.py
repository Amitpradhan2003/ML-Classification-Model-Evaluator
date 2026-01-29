from sklearn.impute import SimpleImputer

class MissingImputation:
    def __init__(self, df):
        self.df = df

    def impute(self):
        num_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = self.df.select_dtypes(include=["object"]).columns

        num_imputer = SimpleImputer(strategy="mean")
        cat_imputer = SimpleImputer(strategy="most_frequent")

        if len(num_cols) > 0:
            self.df[num_cols] = num_imputer.fit_transform(self.df[num_cols])

        if len(cat_cols) > 0:
            self.df[cat_cols] = cat_imputer.fit_transform(self.df[cat_cols])

        print("✅ Missing Values Imputed")
        return self.df
