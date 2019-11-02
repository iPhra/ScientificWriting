import preprocessor as p
import numpy as np
import pandas as pd
import string

import itertools
import collections

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer

from src.utils.menu import yesno_choice

SAMPLE_SIZE = 20000


def main():

    def preprocess(df):

        def laugh_transformer(st):
            # Needs to be enriched
            if "hah" in st or "heh" in st or "hih" in st:
                return "laugh_word"
            else:
                return st

        def negative_transformer(st):
            if st in negative_stop_words:
                return "negative_word"
            else:
                return st

        def clean_tweet(text):
            p.set_options(p.OPT.URL, p.OPT.MENTION)
            ps = PorterStemmer()

            text = p.clean(text)
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = text.strip()

            text = word_tokenize(text)

            # Insert negative_word label for negative words
            text = [negative_transformer(laugh_transformer(ps.stem(w))) for w in text if w not in words_to_remove]

            # removing vowels (coooool -> cool) -> COMMENTED SINCE REDUCING PERFORMANCE
            # text = [remove_vowel(w) for w in text if not w in stop_words]
            # OR: another approach before tokenization tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet))

            # Slang correction
            # TODO getting input file and create dictionary

            # Mispelled word correction -> using PyEnchant (not working on kernels) - bu

            return ' '.join(text)

        tweets_nsw = [word_tokenize(tweet) for tweet in list(df['tweet'])]
        all_words_nsw = list(itertools.chain(*tweets_nsw))

        counts_nsw = collections.Counter(all_words_nsw)  # Is a dict

        singletons = []
        for k in counts_nsw.keys():
            if counts_nsw.get(k) == 1:
                singletons += [k]

        stop_words = set(stopwords.words("english"))  # create a set of stopwords
        stop_words = stop_words.union(["im", "thats"])  # add these stopwords
        negative_stop_words = [w for w in stop_words if "'t" in w]
        negative_stop_words += ["dont", "cant", "dnt", "not"]  # TODO: maybe search for other skewed neg words
        stop_words = stop_words - set(negative_stop_words)  # remove not from the stopwords

        # Finally add to the list of words to be removed
        words_to_remove = []
        words_to_remove += stop_words
        words_to_remove += singletons

        # apply the preprocessing to all rows
        df['tidy_tweet'] = np.vectorize(clean_tweet)(df.tweet)

        df_final = df[['sentiment', 'tidy_tweet']]
        df_final.columns = ["sentiment", "tweet"]

        return df_final




    def create_dataset_sentiment140(n_samples=SAMPLE_SIZE):
        df = pd.read_csv("../datasets/original/Sentiment140.csv", encoding="ISO-8859-1",
                         names=["sentiment", "id", "date", "query", "user", "tweet"])

        if n_samples < df.shape[0]:
            df = df.sample(n_samples)  # extract 20k random samples and make a new dataframe
            df = df.reset_index(drop=True)  # reset the index for all rows

        # Encode sentiment
        df.loc[df.sentiment == 4, 'sentiment'] = 1

        df = preprocess(df)

        df.to_csv("../datasets/preprocessed/Sentiment140.csv", index=False)



    def create_dataset_airlines(n_samples=SAMPLE_SIZE):
        data = pd.read_csv("../datasets/original/Airline.csv", encoding="ISO-8859-1")

        df = data[["airline_sentiment", "text"]]
        df.columns = ["sentiment", "tweet"]

        if n_samples < df.shape[0]:
            df = df.sample(n_samples)  # extract 20k random samples and make a new dataframe
            df = df.reset_index(drop=True)  # reset the index for all rows

        # Remove neutral sentiments and encode
        df = df[df["sentiment"] != "neutral"]
        df.loc[df["sentiment"] == "positive", 'sentiment'] = 1
        df.loc[df["sentiment"] == "negative", 'sentiment'] = 0

        df = preprocess(df)

        df.to_csv("../datasets/preprocessed/airline.csv", index=False)



    yesno_choice('Do you want to create the Sentiment140 Dataset?', callback_yes=(lambda: create_dataset_sentiment140()))
    print("Done datasets Sentiment140")

    yesno_choice('Do you want to create the Airline Dataset?', callback_yes=(lambda: create_dataset_airlines()))
    print("Done datasets Airline")




if __name__ == '__main__':
    main()
