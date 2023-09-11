#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


import pandas as pd
import setUpConnectionToDB as db_conn


def connectToDB() -> object:
    """Set up engine for database connection"""
    engine, connection = db_conn.setUpEngineDB()
    return engine, connection


def queryData(connection) -> pd.DataFrame:
    """query data to evaluate maximum and minimum"""
    query_max = f"""SELECT * 
                    FROM "calendar"
                 """
    return pd.DataFrame(connection.execute(query_max).fetchall())


def main():
    engine, connection = connectToDB()
    df = queryData(connection)


if __name__ == "__main__":
    main()
