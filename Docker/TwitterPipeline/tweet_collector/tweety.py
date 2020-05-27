#https://realpython.com/twitter-bot-python-tweepy/
import pandas as pd
import tweepy
from config import Twitter # storage for authentification
import time

print(Twitter)
## app only authentification
consumer_key = Twitter['consumer_key']
consumer_secret = Twitter['consumer_secret']
access_token = Twitter['access_token']
access_token_secret = Twitter['access_token_secret']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

## user authentification
auth.set_access_token(Twitter['access_token'], Twitter['access_token_secret'])

## access to REST Api (no streaming)
twitterAPI = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

wd = twitterAPI.get_user('W_D')   # 'W_D': use your person of interest instead
df_WD = pd.DataFrame(columns=["Tweet Text", "Tweet Created On"])

def get_tweets(readUser):
    point = tweepy.Cursor(twitterAPI.user_timeline, id=readUser)
    counter = 0
    while(tweepy.Cursor(twitterAPI.user_timeline).items() != None) and counter < 3:
        #counter += 1
        try:
            for page in point.pages():
                for tweet in page:
                    counter += 1
                    if counter > 3:
                        return
                    yield tweet
        except tweepy.RateLimitError:
            print("You reached the rate limit! Wait 15 minutes to proceed...")
            #time.sleep(60*15)

retrieved_data = get_tweets(wd.id)

count_tweets = 0
for tweet in retrieved_data:
    print(tweet.text)
    print("Posted on: ", tweet.created_at)
    df_WD.loc[count_tweets, "Tweet as text"] = tweet.text
    df_WD.loc[count_tweets, "Tweet posted on"] = tweet.created_at

df_WD.to_csv("twitterData.csv")

print(retrieved_data)
print('\n\n****************************')
print('Finished!!!')
