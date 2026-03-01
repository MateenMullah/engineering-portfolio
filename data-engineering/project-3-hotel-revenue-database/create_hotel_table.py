import psycopg

CONN_INFO = "host=localhost port=5432 dbname=hotel_db user=mateen password=mateen_pw"

with psycopg.connect(CONN_INFO) as conn:
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS meal_cost (
            meal TEXT PRIMARY KEY,
            cost DOUBLE PRECISION NOT NULL
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS market_segment_discount (
            market_segment TEXT PRIMARY KEY,
            discount DOUBLE PRECISION NOT NULL
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS bookings (
            id BIGSERIAL PRIMARY KEY,

            -- core identifiers / categories
            hotel TEXT,
            arrival_date_year INT,
            arrival_date_month TEXT,
            arrival_date_week_number INT,
            arrival_date_day_of_month INT,

            -- stay info
            stays_in_weekend_nights INT,
            stays_in_week_nights INT,

            -- people
            adults INT,
            children INT,
            babies INT,

            -- business fields
            meal TEXT,
            country TEXT,
            market_segment TEXT,
            distribution_channel TEXT,
            is_canceled INT,

            -- money / pricing
            adr DOUBLE PRECISION,

            -- status
            reservation_status TEXT,
            reservation_status_date DATE
        );
        """)

        # 3) Drop constraints if they exist (so script can be re-run safely) chatGPT gave me this to fix my code im kinda confused but it works so YAY
        cur.execute("""
        ALTER TABLE bookings
        DROP CONSTRAINT IF EXISTS fk_bookings_meal;
        """)

        cur.execute("""
        ALTER TABLE bookings
        DROP CONSTRAINT IF EXISTS fk_bookings_market_segment;
        """)

        # foreign key says this column must match a key in another table 
        # foreign key ensures meal column in bookings table can only contain values that exits in meal_cost table (HB BB FB SC
        cur.execute("""
        ALTER TABLE bookings
        ADD CONSTRAINT fk_bookings_meal
        FOREIGN KEY (meal) REFERENCES meal_cost(meal);
        """)

        cur.execute("""
        ALTER TABLE bookings
        ADD CONSTRAINT fk_bookings_market_segment
        FOREIGN KEY (market_segment) REFERENCES market_segment_discount(market_segment);
        """)

        conn.commit()
        print("Tables + constraints created successfully.")