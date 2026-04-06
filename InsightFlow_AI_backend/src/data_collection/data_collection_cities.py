import pandas as pd

def load_cities():
    df = pd.read_csv("cities.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    return df  # used by normalisation agent, not standalone