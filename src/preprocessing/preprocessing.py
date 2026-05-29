class TransformerFaultPreprocessor:
    def __init__(self, text_columns):
        self.text_columns = text_columns

    def transform(self, df):
        return df
