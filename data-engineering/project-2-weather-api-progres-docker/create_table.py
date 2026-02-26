import psycopg

CONN_INFO = "host=localhost port=5432 dbname=weather_db user=mateen password=mateen_pw"

with psycopg.connect(CONN_INFO) as conn:
    with conn.cursor() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS weather_hourly (time TIMESTAMP PRIMARY KEY,date DATE,time_of_day TIME,temperature_2m_c DOUBLE PRECISION);")
        conn.commit()
        print("Table created successfully")