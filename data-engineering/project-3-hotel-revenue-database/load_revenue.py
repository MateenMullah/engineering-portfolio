import psycopg

CONN_INFO = "host=localhost port=5432 dbname=hotel_db user=mateen password=mateen_pw"

city_rev_2018 = 0.0
city_rev_2019 = 0.0
city_rev_2020 = 0.0
resort_rev_2018 = 0.0
resort_rev_2019 = 0.0
resort_rev_2020 = 0.0

with psycopg.connect(CONN_INFO) as conn:
    with conn.cursor() as cur:
        # Pull only what we need from the DB
        cur.execute("""
            SELECT
                hotel,
                arrival_date_year,
                stays_in_week_nights,
                stays_in_weekend_nights,
                adr
            FROM bookings;
        """)

        rows = cur.fetchall()

        for row in rows:
            hotel_type, year, week_nights, weekend_nights, adr = row

            # Handle NULLs safely
            if week_nights is None:
                week_nights = 0
            if weekend_nights is None:
                weekend_nights = 0
            if adr is None:
                adr = 0.0

            nights = week_nights + weekend_nights
            revenue = nights * adr

            # Accumulate per hotel + year
            if year == 2018:
                if hotel_type == "City Hotel":
                    city_rev_2018 += revenue
                elif hotel_type == "Resort Hotel":
                    resort_rev_2018 += revenue

            elif year == 2019:
                if hotel_type == "City Hotel":
                    city_rev_2019 += revenue
                elif hotel_type == "Resort Hotel":
                    resort_rev_2019 += revenue

            elif year == 2020:
                if hotel_type == "City Hotel":
                    city_rev_2020 += revenue
                elif hotel_type == "Resort Hotel":
                    resort_rev_2020 += revenue

        cur.execute("TRUNCATE TABLE hotel_revenue;")

        summary_rows = [
            ("City Hotel", 2018, city_rev_2018),
            ("City Hotel", 2019, city_rev_2019),
            ("City Hotel", 2020, city_rev_2020),
            ("Resort Hotel", 2018, resort_rev_2018),
            ("Resort Hotel", 2019, resort_rev_2019),
            ("Resort Hotel", 2020, resort_rev_2020),
        ]

        cur.executemany(
            """
            INSERT INTO hotel_revenue (hotel_type, year, revenue)
            VALUES (%s, %s, %s);
            """,
            summary_rows
        )

    conn.commit()

print("hotel_revenue table updated.")
print("City:", city_rev_2018, city_rev_2019, city_rev_2020)
print("Resort:", resort_rev_2018, resort_rev_2019, resort_rev_2020)