import re
import twint
import pandas as pd
from textblob import TextBlob
from textblob.en.sentiments import PatternAnalyzer
from textblob.en.taggers import PatternTagger
from src.others import nlp_pipeline


def get_csv():
    tw = twint.Config()
    tw.Search ="PlusDe70kgEtSereine"
    tw.Since = "2020-07-27 14:00:00"
    tw.Custom["tweet"] = ["id"]
    tw.Pandas = True
    tw.Lang = "fr"
    twint.run.Search(tw)
    tweets = twint.storage.panda.Tweets_df
    tweets.to_csv('tweet_hashtag.csv')

    def get_tweet_sentiment(tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(tweet)
        # set sentiment
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'


get_csv()
tweet = pd.read_csv("tweet_hashtag.csv")
corpus = tweet['tweet']
corpus_clean = corpus.apply(nlp_pipeline)


polarity = []
for tweet in corpus_clean:
    polarity.append(TextBlob(tweet,pos_tagger=PatternTagger(),analyzer=PatternAnalyzer()).sentiment[0])


# picking positive tweets from tweets
ptweets = [tweet for tweet in corpus if TextBlob(tweet).polarity > 0]
# percentage of positive tweets
print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(corpus)))
# picking negative tweets from tweets
ntweets = [tweet for tweet in corpus if TextBlob(tweet).polarity < 0]
# percentage of negative tweets
print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(corpus)))
# percentage of neutral tweets
print("Neutral tweets percentage: {} % \
      ".format(100*(len(corpus) -(len( ntweets )+len( ptweets)))/len(corpus)))

# printing first 5 positive tweets
print("\n\nPositive tweets:")
for i in range(3):
    print(ptweets[i])

# printing first 5 negative tweets
print("\n\nNegative tweets:")
for i in range(3):
    print(ntweets[i])