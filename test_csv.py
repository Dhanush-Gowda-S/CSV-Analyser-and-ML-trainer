import pandas as pd
file_path = "C:\\BDA\\static\\uploads\\2_e6679d89_8dfb982945d44add88fad184617e0a7c_driverresponse.csv"

print("\n===== Loading CSV =====")
df = pd.read_csv(file_path)
print("Loaded:", file_path)
print("Rows:", len(df))
print("Columns:", len(df.columns))

df.columns = (
    df.columns.str.strip()
              .str.replace(" ", "_")
              .str.replace(r"[^\w_]", "", regex=True)
)

print("\n===== Cleaned Column Names =====")
for c in df.columns:
    print(" -", c)


print("\n===== COLUMN DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values")


print("\n===== SUMMARY (NUMERIC) =====")
print(df.describe())

print("\n===== CORRELATIONS =====")
print(df.corr(numeric_only=True))

numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
string_cols  = df.select_dtypes(include=['object']).columns.tolist()

print("\n===== COLUMN ANALYSIS =====")
print("Numeric Columns (usable for regression):")
for c in numeric_cols:
    print(" -", c)

print("\nString Columns (need OneHotEncoding for ML):")
for c in string_cols:
    print(" -", c)


print("\n===== ML TARGET SUGGESTIONS =====")
print("Possible Regression Targets:")
for c in numeric_cols:
    if "id" not in c.lower() and "index" not in c.lower():
        print(" -", c)

print("\nYou can use this file for ML training.")
print("✔ Use numeric columns directly")
print("✔ Use string columns with OneHotEncoder")
