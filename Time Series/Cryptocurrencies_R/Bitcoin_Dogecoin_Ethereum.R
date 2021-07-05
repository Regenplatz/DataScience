# set working directory to the path where Elon Musk's Tweets (as csv) files are located
setwd("")

# import libraries
library(Dict)
library(quantmod)
library(dplyr)
library(tidyverse)
library(zoo)
library(fBasics)
library(lmtest)
library(tseries)
library(gtrendsR)
library(vars)


#----------------------------------------------------#
#------------Dogecoin, Bitcoin, Ether----------------#
#-----------Data Receipt and Preprocessing-----------#
#----------------------------------------------------#

#------Downloading prizes via Yahoo Finance API------#

data <- NULL
cryptos <- c("BTC-USD", "DOGE-USD", "ETH-USD")
crypto_dict <- dict(x=0)

for (Ticker in cryptos){
  data <- cbind(data,
                getSymbols.yahoo(Ticker, from="2016-06-01", periodicity = "daily", auto.assign=FALSE)[,6])
  data2 <- na.locf(data, fromLast=TRUE)
  data2 <- as.data.frame(data2)
  data2 <- rownames_to_column(data2, "date")
  data2[,"date"] <- as.Date(data2[,"date"], format="%Y-%m-%d")
  crypto_dict[Ticker] <- data2
}


#---Assign corresponding data to crypto variables----#

# Bitcoin
bitcoin <- (crypto_dict[cryptos[1]])
bitcoin_ts <- xts(bitcoin$BTC.USD.Adjusted, bitcoin[,1], start=as.numeric(format(as.Date("2016-06-01"), "%j")))
colnames(bitcoin_ts) <- "BTC.USD.Adjusted"

# Dogecoin
dogecoin <- crypto_dict[cryptos[2]]
dogecoin_ts <- xts(dogecoin$DOGE.USD.Adjusted, dogecoin[,1], start=as.numeric(format(as.Date("2016-06-01"), "%j")))
colnames(dogecoin_ts) <- "DOGE.USD.Adjusted"

# Ether
ether <- crypto_dict[cryptos[3]]
ether_ts <- xts(ether$ETH.USD.Adjusted, ether[,1], start=as.numeric(format(as.Date("2016-06-01"), "%j")))
colnames(ether_ts) <- "ETH.USD.Adjusted"


#-------Plot all three asset returns together-------#

plot(cbind(bitcoin_ts/1000, dogecoin_ts/1000, ether_ts/1000), main="Crytpocurrencies", xlab="Time", ylab="*10^3 USD")




#----------------------------------------------------#
#---------------Elon Musk's Tweets-------------------#
#-----------Data Receipt and Preprocessing-----------#
#----------------------------------------------------#

# Musk's tweets were downloaded as yearly csv files from 
# https://www.kaggle.com/ayhmrba/elon-musk-tweets-2010-2021 in May 2021

# read Elon Musk's yearly (2016-2021) tweets as csv files into R
tweets <- read.csv("2016.csv")
years <- c(2017, 2018, 2019, 2020, 2021)
for (year in years) {
  tweet <- read.csv(sprintf("%s.csv", year))
  tweets <- rbind(tweets, tweet)
} 

# convert date column to date class
tweets$date <- as.Date(tweets$date, format="%Y-%m-%d")

# select only oberservation from latest 2016-06-01 
# as well as only the date and tweet columns
tweets <- tweets[tweets$date >= "2016-06-01", c("date", "tweet")]

#convert content of tweet column to lowercase letters
tweets$tweet <- tolower(tweets$tweet)

# select tweets in which certain keywords appear
func_crypto_tweets <- function(word1, word2){
  crypto_tweets <- tweets[grepl(sprintf("(%s | %s)", word1, word2), tweets$tweet),]
  return(crypto_tweets)
}

tweets_bitcoin <- func_crypto_tweets("bitcoin", "btc")
tweets_dogecoin <- func_crypto_tweets("dogecoin", "doge")
tweets_ethereum <- func_crypto_tweets("etherium", "ethereum")




#----------------------------------------------------#
#--------Musk's Tweets on Stock Market prizes--------#
#---------Separate Plots per Cryptocurrency---------#
#----------------------------------------------------#

# Assign crypto prizes to the Musk's tweet data of the corresponding day
# and plot the data
par(mfrow=c(1,3))

func_tweets_prizes <- function(prizes, crypto_value, tweets_crypto, ts_prizes, ts_tweet, title) {
  prizes$tweet <- 0
  for (tweet in tweets_crypto$date){    #stimmt nicht, da tweets_bitcoin ersetzt werden muss
    prizes$tweet[prizes$date == tweet] <- crypto_value
  }
  ts_tweet <- xts(prizes$tweet, prizes[,1], start=as.numeric(format(as.Date("2016-06-01"), "%j")))
  print(plot(cbind(ts_prizes, ts_tweet), main=title, ylab="USD"))
  return(ts_tweet)
}

bitcoin_tweet_ts <- func_tweets_prizes(bitcoin, max(bitcoin$BTC.USD.Adjusted), tweets_bitcoin, bitcoin_ts, bitcoin_tweet_ts, "Bitcoin")
dogecoin_tweet_ts <- func_tweets_prizes(dogecoin, max(dogecoin$DOGE.USD.Adjusted), tweets_dogecoin, dogecoin_ts, dogecoin_tweet_ts, "Dogecoin")
ether_tweet_ts <- func_tweets_prizes(ether, max(ether$ETH.USD.Adjusted), tweets_ethereum, ether_ts, ether_tweet_ts, "Ethereum")




#----------------------------------------------------#
#----------Musk's Tweets on Cryptocurrencies---------#
#--Assign/Summarize the Values of the Crypto prizes--#
#-------of the Days Following on Musk's Tweets-------#
#----------------------------------------------------#

# subselect Musk's tweets: 
# generate two groups: One with a defined lag after Musk's Tweets
# the other group for the time between the tweets

# define lag size of interest (in days)
diff_lag <- 2


func_tweet_groups <- function(ts_object, col_name, prize_tweet_ts){
  
  # Create new column with lag values and shift it backwards (diff_lag rows) so that e.g. the 
  # lag data are assigned to the day of the tweet (and not diff_lag days after the tweet)
  lag_index <- index(ts_object) - diff_lag      # create Index, diff_lag days backdated
  lag_xts <- xts(ts(ts_object), lag_index)      # remove date index with ts() and assign lag_xts as new index
  #ts_object$lag <- lag_xts$BTC.USD.Adjusted            # assign backdated values as lag
  ts_object$lag <- lag_xts[,col_name]            # assign backdated values as lag
  ts_object$diff_log <- log(ts_object$lag) - log(ts_object[,col_name])
  ts_object$cdiff <- ts_object$lag - ts_object[,col_name]
  
  # merge asset returns with tweets
  tweet_merged <- merge(ts_object, prize_tweet_ts, join="outer")
  tweet_merged <- na.omit(tweet_merged)
  
  # initiate vectors for time 
  # within tweet lag (tweet_lag) and for the time without tweet (no_tweet)
  tweet_lag <- NULL
  no_tweet <- NULL
  
  # Assign respective data to the separate time series objects:
  # within tweet lag (tweet_lag) and for the time without tweet (no_tweet)
  cnt <- 0
  for (i in 1:dim(tweet_merged)[1]) {
    if (tweet_merged$prize_tweet_ts[i] > 0) {
      cnt <- diff_lag
    }
    elem_xts <- xts(tweet_merged$diff_log[i], index(tweet_merged[i]))
    if (cnt > 0) { 
      if (cnt == diff_lag){
        if (is.null(tweet_lag)) {
          tweet_lag <- elem_xts
        }
        else {
          tweet_lag <- rbind(tweet_lag, elem_xts)
        }
      }
    }
    else {
      if (is.null(no_tweet)) {
        no_tweet <- elem_xts
      }
      else {
        no_tweet <- rbind(no_tweet, elem_xts)
      }
    }
    cnt <- cnt - 1
  }
  #result <- list(tweet_lag, no_tweet)
  return(list(tweet_lag, no_tweet))
}


bitcoin_result <- func_tweet_groups(bitcoin_ts, "BTC.USD.Adjusted", bitcoin_tweet_ts)
btc_tweet_lag <- bitcoin_result[[1]]
btc_no_tweet <- bitcoin_result[[2]]

dogecoin_result <- func_tweet_groups(dogecoin_ts, "DOGE.USD.Adjusted", dogecoin_tweet_ts)
doge_tweet_lag <- dogecoin_result[[1]]
doge_no_tweet <- dogecoin_result[[2]]

ether_result <- func_tweet_groups(ether_ts, "ETH.USD.Adjusted", ether_tweet_ts)
eth_tweet_lag <- ether_result[[1]]
eth_no_tweet <- ether_result[[2]]



#----------------------------------------------------#
#----------------Statistical Analysis----------------#
#----------Musk's Tweets on Cryptocurrencies---------#
#---Evaluate potential significant differences of----#
#-----of the Days Following on Musk's Tweets and-----#
#---------------the time without tweets--------------#
#----------------------------------------------------#

func_tweets_statistics <- function(title, prizes_tweets, prizes_noTweets){
  
  print(sprintf("---------------- %s ----------------", title))
  
  # Check for normal distribution
  print(shapiro.test(as.vector(prizes_tweets)))
  print(shapiro.test(as.vector(prizes_noTweets)))
  
  # t-test
  print(t.test(as.vector(prizes_tweets), as.vector(prizes_noTweets))) 
  
  # Wilcoxon rank-sum test
  print(wilcox.test(as.vector(prizes_tweets), as.vector(prizes_noTweets))) 
}

func_tweets_statistics("Bitcoin", btc_tweet_lag, btc_no_tweet)
func_tweets_statistics("Dogecoin", doge_tweet_lag, doge_no_tweet)
func_tweets_statistics("Ethereum", eth_tweet_lag, eth_no_tweet)

 


#----------------------------------------------------#
#-------------------Google Trends--------------------#
#-----------Data Receipt and Preprocessing-----------#
#----------------------------------------------------#

func_getGoogleTrends <- function(keywords, prize_ts, title){
  
  # use gtrends package to retrieve google trends on the respective keywords given
  crypto_google <- gtrends(keyword=keywords, time="2016-06-01 2021-05-24", geo="")
  google <- as.data.frame(crypto_google$interest_over_time[c("date","hits")])
  google$date <- as.Date(google$date, format="%Y-%m-%d")
  
  # assign values below 1 to 0
  google$hits[google$hits == "<1"] <- 0
  google$hits <- as.numeric(google$hits)
  
  # combine data for different keywords to single dates and recalculate percentage
  google <- google %>%
    group_by(date) %>%
    summarise(hits=sum(hits))
  google$hits <- google$hits / max(google$hits) * 100
  google_ts <- xts(google$hits, google$date)
  
  # Plot Google Trends together with Asset Return 
  par(mfrow=c(2,1))
  print(plot(prize_ts, main=sprintf("%s Asset Return", title), ylab="USD"))
  print(plot(google_ts, main=sprintf("Google Trends for %s", title), ylab="Relative Percentage"))

  return (google_ts)
}

btc_keywords <- c("Bitcoin", "bitcoin", "btc", "BTC", "BITCOIN")
doge_keywords <- c("Dogecoin", "dogecoin", "doge", "DOGE", "DOGECOIN")
eth_keywords <- c("Ethereum", "ethereum", "ETHEREUM", "Etherium",  "ETHERIUM")

btc_google_ts <- func_getGoogleTrends(btc_keywords, bitcoin_ts, "Bitcoin")
doge_google_ts <- func_getGoogleTrends(doge_keywords, dogecoin_ts, "Dogecoin")
eth_google_ts <- func_getGoogleTrends(eth_keywords, ether_ts, "Ethereum")




#----------------------------------------------------#
#----------------Statistical Analysis----------------#
#----------------------------------------------------#
#----------------------------------------------------#


#----------------------Summaries---------------------#

summary(bitcoin)
summary(dogecoin)
summary(ether)


#--Check for Stationarity (Augmented Dickey-Fuller Test)--#

func_adf <- function(data){
  if (adf.test(data)$p.value > 0.05){
    data <- na.omit(diff(log(data)))
  }
  print(adf.test(data))
  return(data)
}

# Cryptocurrency prizes
bitcoin_log_diff <- func_adf(bitcoin_ts)
dogecoin_log_diff <- func_adf(dogecoin_ts) 
ether_log_diff <- func_adf(ether_ts)    

# Musk's tweets
func_adf(bitcoin_tweet_ts)
func_adf(dogecoin_tweet_ts)
func_adf(ether_tweet_ts)

# Google trends
btc_google_log_diff <- func_adf(btc_google_ts)
doge_google_log_diff <- func_adf(doge_google_ts)
eth_google_log_diff <- func_adf(eth_google_ts)
  



#----------------------------------------------------#
#----------------Statistical Analysis----------------#
#-------Cryptocurrency prizes vs Google Trends-------#
#-----------------------and--------------------------#
#-------Cryptocurrency prizes vs Musk's Tweets-------#
#----------------------------------------------------#

func_VAR <- function(title, prizes_logDiff, trends_or_tweets_logDiff, prizes_column){
  
  print(sprintf("------------------ %s ------------------", title))
  
  # Estimate Vector Autoregressive Model (VAR)
  VAR_crypto_google <- VAR(merge(prizes_logDiff, trends_or_tweets_logDiff, join="inner"), ic="AIC")
  print(coeftest(VAR_crypto_google))
  
  # Check for Causality (Granger)
  print(causality(VAR_crypto_google, cause=prizes_column)["Granger"])
  print(causality(VAR_crypto_google, cause="trends_or_tweets_logDiff")["Granger"])
  
  # Degression: Plotting impulse response functions
  print(plot(irf(VAR_crypto_google, impulse=prizes_column, response="trends_or_tweets_logDiff")))
  print(plot(irf(VAR_crypto_google, impulse="trends_or_tweets_logDiff", response=prizes_column)))
}

# Prizes vs Google Trends
func_VAR("Bitcoin Prizes vs Google Trends", bitcoin_log_diff, btc_google_log_diff, "BTC.USD.Adjusted")
func_VAR("Dogecoin Prizes vs Google Trends", dogecoin_log_diff, doge_google_log_diff, "DOGE.USD.Adjusted")
func_VAR("Ethereum Prizes vs Google Trends", ether_log_diff, eth_google_log_diff, "ETH.USD.Adjusted")

# Prizes vs Musk's Tweets
func_VAR("Bitcoin Prizes vs Musk's Tweets", bitcoin_log_diff, bitcoin_tweet_ts, "BTC.USD.Adjusted")
func_VAR("Dogecoin Prizes vs Musk's Tweets", dogecoin_log_diff, dogecoin_tweet_ts, "DOGE.USD.Adjusted")
func_VAR("Ethereum Prizes vs Musk's Tweets", ether_log_diff, ether_tweet_ts, "ETH.USD.Adjusted")



