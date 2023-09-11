#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


import time
import setUpConnectionToDB as db_conn


def createTableCalendar(connection):
    """
    Create table in database for calendar data and assign holidays.
    :param connection:
    :return:
    """
    ## drop table
    stmt_01 = f"""DROP TABLE if exists calendar"""
    connection.execute(stmt_01)

    ## create table
    stmt_02 = f"""CREATE TABLE calendar (
                      date_dim_id              INT NOT NULL,
                      date_actual              DATE NOT NULL,
                      epoch                    BIGINT NOT NULL,
                      day_suffix               VARCHAR(4) NOT NULL,
                      day_name                 VARCHAR(9) NOT NULL,
                      day_of_week              INT NOT NULL,
                      day_of_month             INT NOT NULL,
                      day_of_quarter           INT NOT NULL,
                      day_of_year              INT NOT NULL,
                      week_of_month            INT NOT NULL,
                      week_of_year             INT NOT NULL,
                      week_of_year_iso         CHAR(10) NOT NULL,
                      month_actual             INT NOT NULL,
                      month_name               VARCHAR(9) NOT NULL,
                      month_name_abbreviated   CHAR(3) NOT NULL,
                      quarter_actual           INT NOT NULL,
                      quarter_name             VARCHAR(9) NOT NULL,
                      year_actual              INT NOT NULL,
                      first_day_of_week        DATE NOT NULL,
                      last_day_of_week         DATE NOT NULL,
                      first_day_of_month       DATE NOT NULL,
                      last_day_of_month        DATE NOT NULL,
                      first_day_of_quarter     DATE NOT NULL,
                      last_day_of_quarter      DATE NOT NULL,
                      first_day_of_year        DATE NOT NULL,
                      last_day_of_year         DATE NOT NULL,
                      mmyyyy                   CHAR(6) NOT NULL,
                      mmddyyyy                 CHAR(10) NOT NULL,
                      weekend_indr             BOOLEAN NOT null,
                      working_day			   FLOAT not null 
                    )"""
    connection.execute(stmt_02)

    ## wait for table being set up before starting to insert data (otherwise error)
    time.sleep(1)

    ## add primary key
    stmt_03 = f"""ALTER TABLE public.calendar 
                  ADD CONSTRAINT calendar_date_dim_id_pk 
                  PRIMARY KEY (date_dim_id)"""
    connection.execute(stmt_03)

    ## create index
    stmt_04 = f"""CREATE INDEX calendar_date_actual_idx
                  ON calendar(date_actual)"""
    connection.execute(stmt_04)

    ## insert calendar data into table
    stmt_05 = f"""
                INSERT INTO calendar
                SELECT TO_CHAR(datum, 'yyyymmdd')::INT AS date_dim_id,
                       datum AS date_actual,
                       EXTRACT(EPOCH FROM datum) AS epoch,
                       TO_CHAR(datum, 'fmDDth') AS day_suffix,
                       TO_CHAR(datum, 'TMDay') AS day_name,
                       EXTRACT(ISODOW FROM datum) AS day_of_week,
                       EXTRACT(DAY FROM datum) AS day_of_month,
                       datum - DATE_TRUNC('quarter', datum)::DATE + 1 AS day_of_quarter,
                       EXTRACT(DOY FROM datum) AS day_of_year,
                       TO_CHAR(datum, 'W')::INT AS week_of_month,
                       EXTRACT(WEEK FROM datum) AS week_of_year,
                       EXTRACT(ISOYEAR FROM datum) || TO_CHAR(datum, '"-W"IW-') || EXTRACT(ISODOW FROM datum) AS week_of_year_iso,
                       EXTRACT(MONTH FROM datum) AS month_actual,
                       TO_CHAR(datum, 'TMMonth') AS month_name,
                       TO_CHAR(datum, 'Mon') AS month_name_abbreviated,
                       EXTRACT(QUARTER FROM datum) AS quarter_actual,
                       CASE
                           WHEN EXTRACT(QUARTER FROM datum) = 1 THEN 'First'
                           WHEN EXTRACT(QUARTER FROM datum) = 2 THEN 'Second'
                           WHEN EXTRACT(QUARTER FROM datum) = 3 THEN 'Third'
                           WHEN EXTRACT(QUARTER FROM datum) = 4 THEN 'Fourth'
                           END AS quarter_name,
                       EXTRACT(YEAR FROM datum) AS year_actual,
                       datum + (1 - EXTRACT(ISODOW FROM datum))::INT AS first_day_of_week,
                       datum + (7 - EXTRACT(ISODOW FROM datum))::INT AS last_day_of_week,
                       datum + (1 - EXTRACT(DAY FROM datum))::INT AS first_day_of_month,
                       (DATE_TRUNC('MONTH', datum) + INTERVAL '1 MONTH - 1 day')::DATE AS last_day_of_month,
                       DATE_TRUNC('quarter', datum)::DATE AS first_day_of_quarter,
                       (DATE_TRUNC('quarter', datum) + INTERVAL '3 MONTH - 1 day')::DATE AS last_day_of_quarter,
                       TO_DATE(EXTRACT(YEAR FROM datum) || '-01-01', 'YYYY-MM-DD') AS first_day_of_year,
                       TO_DATE(EXTRACT(YEAR FROM datum) || '-12-31', 'YYYY-MM-DD') AS last_day_of_year,
                       TO_CHAR(datum, 'mmyyyy') AS mmyyyy,
                       TO_CHAR(datum, 'mmddyyyy') AS mmddyyyy,
                       CASE
                           WHEN EXTRACT(ISODOW FROM datum) IN (6, 7) THEN TRUE
                           ELSE FALSE
                           END AS weekend_indr,
                       CASE
                           WHEN EXTRACT(ISODOW FROM datum) IN (6, 7) THEN 0
                           ELSE 1
                           END AS working_day
                FROM (SELECT '2017-01-01'::DATE + SEQUENCE.DAY AS datum
                      FROM GENERATE_SERIES(0, 1825) AS SEQUENCE (DAY)
                      GROUP BY SEQUENCE.DAY) DQ
                ORDER BY 1"""
    connection.execute(stmt_05)


    ## set Monday to Friday as working day
    stmt_06 = f"""UPDATE calendar
                  SET working_day = 1 
                  WHERE day_of_week IN (1,2,3,4,5)"""
    connection.execute(stmt_06)


    ## HOLIDAYS: FIXED DATES ----------------------------------------------

    ## New Year (01.Jan.20xy)
    stmt_07 = f"""UPDATE calendar
                  SET working_day = 0 
                  WHERE month_actual = 1
                    AND day_of_month IN (1)"""
    connection.execute(stmt_07)

    ## Labor Day (01.May.20xy)
    stmt_08 = f"""UPDATE calendar
                  SET working_day = 0 
                  WHERE month_actual = 5
                    AND day_of_month IN (1)"""
    connection.execute(stmt_08)

    ## Swiss National Day (01.Aug.20xy)
    stmt_09 = f"""UPDATE calendar
                  SET working_day = 0 
                  WHERE month_actual = 8
                    AND day_of_month IN (1)"""
    connection.execute(stmt_09)

    ## Christmas (25.& 26.Dec.20xy)
    stmt_10 = f"""UPDATE calendar
                  SET working_day = 0 
                  WHERE month_actual = 12
                    AND day_of_month IN (25, 26)"""
    connection.execute(stmt_10)


    ## HOLIDAYS: FLEXIBLE DATES ----------------------------------------------

    ## Good Friday, Easter Monday, Ascension Day, Whit Monday
    stmt_11 = f"""UPDATE calendar
                  SET working_day = 0 
                  WHERE date_dim_id IN (20170414, 20180330, 20190419, 20200410, 20210326,
                                        20170417, 20180402, 20190422, 20200413, 20210329,
                                        20170525, 20180510, 20190530, 20200521, 20210506,
                                        20170605, 20180521, 20190610, 20200601, 20210517)"""
    connection.execute(stmt_11)

    ## set Fridays after free Thursdays to not a working day
    stmt_12 = f"""UPDATE calendar 
                  SET working_day = 0 
                  WHERE date_dim_id IN (SELECT date_dim_id + 1
                                        FROM calendar
                                        WHERE day_name = 'Thursday' and working_day = 0
                                        ORDER BY date_dim_id)"""
    connection.execute(stmt_12)

    ## set Wednesdays before free Thursdays to a half working day
    stmt_13 = f"""UPDATE calendar 
                  SET working_day = 0.5
                  WHERE date_dim_id IN (SELECT date_dim_id - 1
                                        FROM calendar
                                        WHERE day_name = 'Thursday' and working_day = 0
                                        ORDER BY date_dim_id
                                ) AND working_day = 1"""
    connection.execute(stmt_13)

    ## set primary key
    try:
        stmt_14 = f"""ALTER TABLE calendar 
                       ADD PRIMARY KEY(date_dim_id)
                    """
        connection.execute(stmt_14)
    except:
        pass

    print("CALENDAR table created ------------------------------")


def main():
    """
    Connect to database, create table for calendar data, insert data.
    Set primary key and assign holidays.
    """
    ## Set up engine for database connection
    engine, connection = db_conn.setUpEngineDB()

    ## create table in database and insert data
    createTableCalendar(connection)


if __name__ == "__main__":
    main()
