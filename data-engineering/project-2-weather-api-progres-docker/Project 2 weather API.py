import polars as pl
from pathlib import Path
import requests

def clean_columns(col:str) -> str:
    return (
        col.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("°", "")
        .replace("__", "_")
    )


api_url = "https://api.open-meteo.com/v1/forecast?latitude=-26.2023&longitude=28.0436&hourly=temperature_2m"
response = requests.get(api_url)

#response is a object cannot say response == 200 as "“Is this entire response object equal to the number 200?" no therefore condition fails 
#response.status_code == 200 then everything worked
if response.status_code == 200:
    #response.json() is parsing the JSON text into python as a dictonary 
    data = response.json()
    #print(data)
    #print(type(data))
    print(data.keys())
    print(type(data["hourly"]))
    print(data["hourly"].keys())
    table_data = {
        "time": data["hourly"]["time"],
        #special characters and spaces cause issues
        "temperature_2m (°C)": data["hourly"]["temperature_2m"]
    }
    df = pl.DataFrame(table_data)
    print(df.head())
    df = df.with_columns(
        #strict = False ensures bad parses are turned into nulls (no crash)
    pl.col("time").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M", strict=False)
    )
    #splits date and time column into date column and time column, added to the right of the previous columns 
    df = df.with_columns([
    pl.col("time").dt.date().alias("date"),
    pl.col("time").dt.time().alias("time_of_day")
    ])
    print(df.head())
    print(df.null_count())
    print("Cleaned Columns")
    for c in df.columns.copy():
        new_name = clean_columns(c)
        df = df.rename({c: new_name})
    print(df.head())
    print(df.schema)

    BASE_DIR = Path(__file__).parent
    out_path = BASE_DIR / "weather_API.parquet"
    df.write_parquet(out_path)
    print("Wrote parquet to:", out_path)

else: 
    print("Request failed with status:", response.status_code)
