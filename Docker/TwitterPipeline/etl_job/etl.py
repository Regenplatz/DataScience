"""connect to Mongo database, pull tweets, perform sentiment analysis and load into Postgres"""

import sys
import time
import pymongo
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sqlalchemy import create_engine


##### postgres vars
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "postgres"
HOST = 'postgres_container'
DB = 'postgres'
PORT = '5432'


##### mongodb
client = None
while not client:
    try:
        client = pymongo.MongoClient("mongodb", port=27017)
        db = client.tweets
        table = db.collections.regenplatz
    except:
        time.sleep(2)
        continue


##### postgres
engine = None
while not engine:
    try:
        URI = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{HOST}:{PORT}/{DB}"
        print(URI)
        engine = create_engine(URI, echo=True)
    except:
        print("got error:", sys.exc_info()[0])
        time.sleep(1)
        continue


##### create table with tweets, retweet count and sentiments
time.sleep(1)
engine.execute('''CREATE TABLE IF NOT EXISTS tweets (
    tweets TEXT,
    retweet_count INTEGER,
    sentiment FLOAT(16)    );
    ''')
engine.execute('''DELETE FROM tweets''')


def get_tweet_from_mongo(row_number):
    """Extract tweet from Mongo, or return nothing (exception handling included)."""
    tweet = list(table.find())
    try:
        if tweet:
            tweet2 = tweet[row_number]['text'].replace("'", "")
            retweet_ct = tweet[row_number]['retweet_count']
            return tweet2, retweet_ct
        else:
            return "",""
    except IndexError :
        print(sys.exc_info()[0])
        return "",""


def sentiment(tweet):
    """Evaluate sentiment of a tweet, or return 0.0 as neutral score."""
    sia = SentimentIntensityAnalyzer()
    if tweet:
        sentiment = sia.polarity_scores(tweet)
        return sentiment['compound']
    else:
        return 0.0


def write_to_postgres(tweet, retweet_ct, sentiment):
    """Insert a row of tweet, retweet count and sentiment into Postgres database"""
    engine.execute(f"""INSERT INTO tweets VALUES ('{tweet}', {retweet_ct}, {sentiment});""")


def main():
    """Call functions get_tweet_from_mongo(), sentiment() and write_to_postgres()."""
    time.sleep(10)          # wait for tweety to collect all tweets
    doc_count = 0
    while doc_count == 0:
        doc_count = db.collections.regenplatz.find().count()
        time.sleep(10)
    i=0
    not_eof = True
    while not_eof:
        tweet, retweet_ct = get_tweet_from_mongo(i)
        i += 1
        if tweet == "":
            not_eof = False
        else:
            senti_score = sentiment(tweet)
            write_to_postgres(tweet, retweet_ct, senti_score)


if __name__ == "__main__":
    main()