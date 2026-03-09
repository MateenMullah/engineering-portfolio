import polars as pl 
import psycopg

CONN_INFO = "host=127.0.0.1 port=5432 dbname=weather_db user=mateen password=mateen_pw connect_timeout=5"
path = r"C:\Users\matee\OneDrive\Desktop\skripsie\VScode\DataEngineeringLearning\weather_API.parquet"
csv_path = "weather_load.csv" 

df = pl.read_parquet(path)

#copy column order to match SQL order
df.select(["time", "date", "time_of_day", "temperature_2m_c"]).write_csv(csv_path)
print("CSV written:", csv_path)

with psycopg.connect(CONN_INFO) as conn:
    with conn.cursor() as cur:
        #deletes all rows from weather_hourly (Every run fully replaces the table contents with the latest parquet)
        cur.execute("TRUNCATE weather_hourly;")
        with open(csv_path, "r", encoding="utf-8") as f:
            with cur.copy(
            """
            COPY weather_hourly (time, date, time_of_day, temperature_2m_c)
            FROM STDIN WITH (FORMAT CSV, HEADER TRUE)
            """
            ) as copy:
                copy.write(f.read())
        conn.commit()
        cur.execute("SELECT COUNT(*) FROM weather_hourly;")
        print("Rows in table:", cur.fetchone())


print("Data loaded successfully.")