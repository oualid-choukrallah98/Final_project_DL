import pandas as pd


def clean_findings(df):
    df['findings'] = df['findings'].str.replace('XXXX', '', regex=False)
    df['findings'] = df['findings'].str.replace('xxxx', '', regex=False)
    # Remove extra spaces
    df['findings'] = df['findings'].str.replace(r'\s+', ' ', regex=True).str.strip()
    return df