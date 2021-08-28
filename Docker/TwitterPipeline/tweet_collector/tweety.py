"""Get tweets from Twitter and load it into Mongo database."""

import pandas as pd
import tweepy
import pymongo
from config import Twitter      # storage for authentification
import time


##### connection to Mongo database
conn = 'mongodb'
client = pymongo.MongoClient(conn)
db = client.tweets

##### authentification (Twitter API)
consumer_key = Twitter['consumer_key']
consumer_secret = Twitter['consumer_secret']
access_token = Twitter['access_token']
access_token_secret = Twitter['access_token_secret']

##### user authentification
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(Twitter['access_token'], Twitter['access_token_secret'])

##### access to REST Api (no streaming)
twitterAPI = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
wd = twitterAPI.get_user('W_D')          # 'W_D': use your person of interest instead
df_WD = pd.DataFrame(columns=["Tweet Text", "Tweet Created On"])


def get_tweets(readUser):
    """Get up to 50 tweets of your person of interest from twitter."""
    point = tweepy.Cursor(twitterAPI.user_timeline, id=readUser)
    counter = 0
    while(tweepy.Cursor(twitterAPI.user_timeline).items() != None):
        try:
            for page in point.pages():
                for tweet in page:
                    counter += 1
                    time.sleep(1)
                    if counter >= 50:
                        return
                    yield tweet
        except tweepy.RateLimitError:
            print("You reached the rate limit! Wait 15 minutes to proceed...")
            time.sleep(60*15)


if __name__ == "__main__":
    retrieved_data = get_tweets(wd.id)
    count_tweets = 0
    db.collections.regenplatz.delete_many({})
    for tweet in retrieved_data:
        tweet_dict = {
            'text': tweet.text,
            'created_at': tweet.created_at,
            'retweet_count': tweet.retweet_count
        }
        print(tweet_dict)
        db.collections.regenplatz.insert_one(tweet_dict)
