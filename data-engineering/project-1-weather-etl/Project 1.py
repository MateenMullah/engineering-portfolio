import polars as pl
from pathlib import Path

path = r"C:\Users\matee\OneDrive\Desktop\skripsie\VScode\DataEngineeringLearning\Weather_Dataset.csv"
df = pl.read_csv(path)

def clean_columns(col:str) -> str:
    return (
        col.strip()
        .lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("%", "pct")
        .replace("__", "_")
    )

print("Original Columns:")
for c in df.columns:
    print(c)

print("Cleaned Columns")
for c in df.columns:
    new_name = clean_columns(c)
    df = df.rename({c: new_name})

for c in df.columns:
    print(c)

print(df.head())

df = df.with_columns(
    pl.col("date_time").str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M")
)
#df.schema gives column names and their datatype (mucho importante!)
print(df.schema)
print(df.select("date_time").head(5))

df = df.with_columns([
    pl.col("date_time").dt.date().alias("date"),
    pl.col("date_time").dt.hour().alias("hour"),
])

#checks for missing values 
print("Null counts:")
print(df.null_count())
#removes any row where date_time in null only checks column date_time if other columns are null we are chilling 
df = df.drop_nulls(["date_time"])
print("After drop:", df.shape)

print("Schema before write:")
print(df.schema)

BASE_DIR = Path(__file__).parent
out_path = BASE_DIR / "weather_processed.parquet"

df.write_parquet(out_path)
print("Wrote parquet to:", out_path)
