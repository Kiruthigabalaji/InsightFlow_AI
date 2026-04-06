import pandas as pd


# 🔹 Step 1: Load Excel file (same folder)
def load_cre_dataset():
    file_path = "Real-Estate-Capital-Europe-Sample-CRE-Lending-Data.xlsx"

    try:
        df = pd.read_excel(file_path)
        print(f"📊 Loaded CRE dataset: {len(df)} rows")
        return df

    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None


# 🔹 Step 2: Normalize column names
def normalize_columns(df):
    df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]

    print("\n📌 Columns detected:")
    print(df.columns)

    return df


# 🔹 Step 3: Convert to unified schema
def transform_cre_to_schema(df):
    data = []

    for _, row in df.iterrows():

        location = str(row.get("location", row.get("city", "Unknown")))
        lender = str(row.get("lender", row.get("lender_name", "Unknown")))
        amount = str(row.get("loan_amount", row.get("amount", "Unknown")))
        date = str(row.get("date", ""))

        record = {
            "source": "CRE_Lending",
            "title": f"Lending activity in {location}",
            "content": f"{lender} provided a loan of {amount} in {location}",
            "date": date,
            "link": None,

            # Core fields
            "location": location,
            "entities": [lender],
            "signal": "lending_activity",
            "summary": f"{lender} lent {amount} in {location}",
            "confidence": 1.0
        }

        data.append(record)

    return data


# 🔹 Run
if __name__ == "__main__":
    df = load_cre_dataset()

    if df is not None:
        df = normalize_columns(df)

        print("\n🔍 Preview Data:")
        print(df.head())  # debug

        structured_data = transform_cre_to_schema(df)

        print("\n🎯 SAMPLE OUTPUT\n")

        for d in structured_data[:3]:
            print(d)
            print("-" * 80)

        print(f"\n✅ Total Records Processed: {len(structured_data)}")