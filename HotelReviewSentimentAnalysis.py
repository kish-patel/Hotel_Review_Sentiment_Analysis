# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:33:34 2020

@author: patel
"""

# Import necessary modules
import json
from io import open
from mrjob.job import MRJob
from mrjob.step import MRStep
import re
import math
from collections import Counter
# This package stems words down to their root
from nltk.stem import WordNetLemmatizer




class MRSentimentAnalysis(MRJob):
    
    def configure_args(self):
        super(MRSentimentAnalysis, self).configure_args()
        # File with all stop words comma seperated
        self.add_file_arg('--words', help='Path to stopwords-long.txt')
        # Call the original file again to create an object that stores count of each review type
        self.add_file_arg('--reviews', help='Path to hotelreview.json')
    
    def steps(self):
        return [
                 MRStep(mapper=self.mapper_get_words_in_reviews,
                        mapper_init = self.load_stopwords_and_counter,
                        reducer = self.reducer_count_words_by_Stype),
                 MRStep(mapper=self.mapper_total_number_of_words_per_Stype,
                        reducer=self.reducer_total_number_of_words_per_Stype),
                 MRStep(mapper=self.mapper_number_of_reviews_by_Stype_a_word_appear_in,
                        reducer=self.reducer_word_frequency_in_all_Stypes),
                 MRStep(mapper=self.mapper_calculate_tfidf,
                        reducer=self.reducer_get_top_scores)
                ]
    
    def load_stopwords_and_counter(self):
        
        # Create a list of all the stop words in the file
        self.stop_words = []
        with open('stopwords-long.txt', encoding='utf8', errors='ignore') as file:
            for all_words in file:
                self.stop_words = all_words.split(',')
        
        # Create a counter that counts the total number of reviews by sentiment
        # Positive Sentiment = Rating > 3; Negative Sentiment = Rating <= 3
        # (Key: Sentiment type), (Value: TotalReviews)
        self.cnt = Counter()
        with open("hotelreview.json", encoding='ascii', errors='ignore') as f:
            for line in f:
                rating = json.loads(line)['ratings']['overall']
                if rating > 3:
                    sentiment = '+'
                else:
                    sentiment = '-'
                self.cnt[(sentiment)] += 1

    # Step 1: Get all the words in each review by sentiment type
    def mapper_get_words_in_reviews(self, _, line):
        # Remove all punction marks from each review, convert to lowercase
        review = re.sub('[^a-zA-Z]', ' ', json.loads(line)['text'])
        review = review.lower()
        # Create a list of all words in the review
        review = review.split()
        # Create an instance of WordNetLemmatizer() from nltk.stem package we imported
        wnl = WordNetLemmatizer()
        # Lemmatize each word in the review and remove all the stop words
        # Lemmatize example: bats ==> bat; rocks ==> rock
        review = [wnl.lemmatize(w) for w in review if not w in self.stop_words]
        rating = json.loads(line)['ratings']['overall']
        if rating > 3:
            sentiment = '+'
        else:
            sentiment = '-'
        # Yield each word, review sentiment, and total reviews by sentiment type as the key for each review
        for word in review:
            yield (word, (sentiment), self.cnt[(sentiment)]), 1
    
        
    def reducer_count_words_by_Stype(self, key, value):
        # Sum up all occurences of a word by sentiment type
        # Output: Key -> (word, sentiment, totalreviewsbyStype) Value -> (wordfreq)
        yield (key[0], key[1], key[2]), sum(value)
    
    
    # Step 2: Now we need to count the total number of words for each sentiment type
    def mapper_total_number_of_words_per_Stype(self, key, value):
        # Output: Key -> (sentiment) Value -> (word, wordfreq, totalreviewsbyStype)
         yield key[1], (key[0],value, key[2])
    
    def reducer_total_number_of_words_per_Stype(self, sentiment, words_per_SType):
        # Compute the total number of words per sentiment type
        total = 0
        wordfreq = []
        word = []
        totalreviewsbyStype = []
        for value in words_per_SType:
            total += value[1]
            wordfreq.append(value[1])
            word.append(value[0])
            totalreviewsbyStype.append(value[2])
            
        totalwordsbyStype = [total]*len(word)
    
        # Output: Key -> (word, sentiment, totalreviewsbyStype) Value -> (wordfreq, totalwordsbyStype)
        for value in range(len(word)):
            yield (word[value], sentiment, totalreviewsbyStype[value]), (wordfreq[value], totalwordsbyStype[value])
            
    # Step 3: Count number of times a word appears in either sentiment type (i.e. positive or negative review)
    def mapper_number_of_reviews_by_Stype_a_word_appear_in(self, key, value):
        # Output: Key -> (word) Value -> (sentiment, wordfreq, totalwordsbyStype, totalreviewsbyStype, 1)
        yield key[0], (key[1], value[0], value[1], key[2] , 1)
    
    
    def reducer_word_frequency_in_all_Stypes(self, word, wordValue):
        # Count number of reviews in the data in which the word appears by sentiment type
        total = 0
        sType = []
        wordfreq = []
        totalwordsbyStype = []
        totalreviewsbyStype = []
        
        for value in wordValue:
            total += 1
            sType.append(value[0])
            wordfreq.append(value[1])
            totalwordsbyStype.append(value[2])
            totalreviewsbyStype.append(value[3])
   
        # we need to compute the total numbers of documents in corpus
        dfreq = [total]* len(wordfreq)
        
        # Output: Key -> (word, sentiment, totalreviewsbyStype) Value -> (wordfreq, totalwordsbyStype, dfreq)
        for value in range(len(dfreq)):
            yield (word, sType[value], totalreviewsbyStype[value]), (wordfreq[value], totalwordsbyStype[value], dfreq[value])
            
     
            
    # Step 4: Calculate the TF-IDF Score
    def mapper_calculate_tfidf(self, key, value):
        #TF-IDF -> (wordfreq / totalwordsbyStype) * log_e(totalreviewsbyStype / dfreq)
        tfidf = (value[0] / value[1]) * math.log10(key[2] / value[2])
        # Output: Key -> (tfidf score) Value -> (word, sentiment)
        yield round(float(tfidf),4), (key[0], key[1])
        
    def reducer_get_top_scores(self, tfidf, value):
        # Get top words
        for word, sentiment in value:
            if tfidf > 0.01:
                yield None, "{0},{1},{2}".format(word, sentiment, round(tfidf,4))

if __name__ == '__main__':
    MRSentimentAnalysis.run()
    
# !python HotelReviewSentimentAnalysis.py  --words=stopwords-long.txt --reviews=hotelreview.json hotelreview.json> output.txt
    
