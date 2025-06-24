import pandas as pd
from .cleaner import TextCleaner
from .config import CSV_PATH

class Chunker:
    def __init__(self):
        self.df = pd.read_csv(CSV_PATH).fillna("")
        self.drop_non_semantic()
        self.clean_columns()

    def drop_non_semantic(self):
        drop_cols = [
            'product_id', 'sku', 'image_url', 'asin', 'upc',
            'model_number', 'internal_id', 'timestamp',
            'created_at', 'updated_at'
        ]
        self.df.drop(columns=[c for c in drop_cols if c in self.df.columns], inplace=True)

    def clean_columns(self):
        for col in self.df.columns:
            self.df[col] = self.df[col].apply(TextCleaner.clean)

    def to_chunks(self):
        def row_chunks(row):
            return [f"{col}: {row[col]}" for col in self.df.columns if row[col]]
        self.df['chunks'] = self.df.apply(row_chunks, axis=1)
        return self.df['chunks'].tolist()
