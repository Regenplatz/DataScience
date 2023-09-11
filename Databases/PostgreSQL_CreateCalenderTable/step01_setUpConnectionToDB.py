#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from dotenv import dotenv_values


## credentials and database specifics
config = dotenv_values(".env")
username = config["username"]
password = config["password"]
db_name = "calendar_db"
db_path = f"postgresql://{username}:{password}@localhost:5432/" + db_name


def createDB():
    """
    Connect to postgres and create new database
    :return: NA
    """
    engine = create_engine(db_path)
    if not database_exists(engine.url):
        create_database(engine.url)


def setUpEngineDB():
    """
    Set up engine for PostgreSQL database connection
    :return: engine, engine for database connection
             connection, database connection to postgres
    """
    engine = create_engine(db_path)
    connection = engine.connect()
    return engine, connection


def main():
    """
    Set up database in postgres and create connection to access database via python scripts
    """
    ## connect to postgres and create new database
    createDB()

    ## Set up engine for new database connection
    engine, connection = setUpEngineDB()


if __name__ == "__main__":
    main()
