import pandas as pd 
import psycopg 
from psycopg.rows import tuple_row

CONN_INFO = "host=localhost port=5432 dbname=hotel_db user=mateen password=mateen_pw"

EXCEL_PATH = r"c:\Users\matee\OneDrive\Desktop\skripsie\VScode\DataEngineeringLearning\hotel_revenue_historical_full-2.xlsx"

meal_df = pd.read_excel(EXCEL_PATH, sheet_name="meal_cost")
seg_df = pd.read_excel(EXCEL_PATH, sheet_name="market_segment")

meal_df = meal_df.rename(columns={"Cost": "cost"}).loc[:, ["meal", "cost"]]
seg_df = seg_df.rename(columns={"Discount": "discount"}).loc[:, ["market_segment", "discount"]]

meal_rows = list(meal_df.itertuples(index=False, name=None))
seg_rows = list(seg_df.itertuples(index=False, name=None))

with psycopg.connect(CONN_INFO) as conn:
    with conn.cursor(row_factory=tuple_row) as cur:

# %s is a parameter placeholder (works for all data types), “Insert a value here from the tuple being passed.”
# SET discount = EXCLUDED.discount Update the existing row’s discount to the new value we attempted to insert
        cur.executemany(
            """
            INSERT INTO meal_cost (meal, cost)
            VALUES (%s, %s)
            ON CONFLICT (meal) DO UPDATE
            SET cost = EXCLUDED.cost
            """,
        meal_rows
        )

        cur.executemany(
            """
            INSERT INTO market_segment_discount (market_segment, discount)
            VALUES (%s, %s)
            ON CONFLICT (market_segment) DO UPDATE
            SET discount = EXCLUDED.discount;
            """,
            seg_rows
        )
    conn.commit()

print(f"Loaded {len(meal_rows)} rows into meal_cost and {len(seg_rows)} rows into market_segment_discount.")