import pandas as pd
import psycopg

CONN_INFO = "host=localhost port=5432 dbname=hotel_db user=mateen password=mateen_pw"
EXCEL_PATH = r"c:\Users\matee\OneDrive\Desktop\skripsie\VScode\DataEngineeringLearning\hotel_revenue_historical_full-2.xlsx"

df18 = pd.read_excel(EXCEL_PATH, sheet_name="2018")
df19 = pd.read_excel(EXCEL_PATH, sheet_name="2019")
df20 = pd.read_excel(EXCEL_PATH, sheet_name="2020")

df = pd.concat([df18, df19, df20], ignore_index=True)

cols = [
    "hotel",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_canceled",
    "adr",
    "reservation_status",
    "reservation_status_date",
]
df = df.loc[:, cols]

df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"], errors="coerce").dt.date

num_cols = [
    "arrival_date_year", "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies",
    "is_canceled",
    "adr",
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

text_cols = ["hotel","arrival_date_month","meal","country","market_segment","distribution_channel","reservation_status"]
for c in text_cols:
    df[c] = df[c].astype("string")

df = df.astype(object).where(pd.notnull(df), None)
rows = list(df.itertuples(index=False, name=None))
rows = list(rows)

with psycopg.connect(CONN_INFO) as conn:
    with conn.cursor() as cur:   
        cur.execute("TRUNCATE TABLE bookings RESTART IDENTITY;")

        cur.executemany(
            """
            INSERT INTO bookings (
                hotel, arrival_date_year, arrival_date_month, arrival_date_week_number, arrival_date_day_of_month,
                stays_in_weekend_nights, stays_in_week_nights,
                adults, children, babies,
                meal, country, market_segment, distribution_channel, is_canceled,
                adr, reservation_status, reservation_status_date
            )
            VALUES (%s,%s,%s,%s,%s, %s,%s, %s,%s,%s, %s,%s,%s,%s,%s, %s,%s,%s);
            """,
            rows
        )
    conn.commit()

print(f"Loaded {len(rows)} rows into bookings.")