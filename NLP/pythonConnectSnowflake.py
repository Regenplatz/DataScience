#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


import pandas as pd
import snowflake.connector
from dotenv import dotenv_values
import defineQuery as dq


def lookUpPW(path: str) -> str:
    """Read credentials from .env file"""
    config = dotenv_values(path)
    return config["PASSWORD"]


def queryData(query: str, pw: str) -> pd.DataFrame:
    """Connect to Snowflake, query data, load to dataframe and close connection"""
    conn = snowflake.connector.connect(
        user="<USERNAME>",
        password=pw,
        account="<ACCOUNT>",
        warehouse="<WAREHOUSE>",
        database="<DATABASE>",
        schema="<SCHEMA>"
    )
    cur = conn.cursor()
    cur.execute(query)
    df = cur.fetch_pandas_all()
    conn.close()
    return df


def main():

    ## get password
    path = ".env"
    pw = lookUpPW(path)

    ## query
    query = dq.returnQuery()
    df = queryData(query, pw)
    print(df)


if __name__ == "__main__":
    main()
