"""
Kaito Minami
DS 3500 / Christmas Song NLP Analysis
HW3: Reusable NLP Library
Created: 02/20/2023 | Last Modified: 02/27/2023
"""

# import statements
from collections import Counter, defaultdict, OrderedDict
from functools import reduce
import pandas as pd
import sankey as sk
import matplotlib.pyplot as plt
from statistics import mean
import json
import pprint as pp

# source: https://stackoverflow.com/questions/14694482/converting-html-to-text-with-python
from bs4 import BeautifulSoup

# source: https://realpython.com/python-nltk-sentiment-analysis/
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# downloads stopwords
nltk.download([
    # "names",
    "stopwords",
    # "vader_lexicon",
    # "punkt"
])
sia = SentimentIntensityAnalyzer()


class Txtanalyzer:
    def __init__(self):
        """ manage data about the different texts that
            we register with the framework
        data: statistics of input texts
        counter: count of how many files are loaded to one Txtanalyzer object
        """
        self.data = defaultdict(dict)
        self.counter = int()

    def _save_results(self, results, label):
        """ Integrate parsing results into internal state
        :param label: (str) unique label for a text file that we parsed
        :param results: (dict) the data extracted from the file as a dictionary attribute-->raw data
        :return: updates the data dictionary with results
        """
        assert isinstance(results, dict), 'You chose the wrong data type'
        assert isinstance(label, str), 'You chose the wrong data type'

        for k, v in results.items():
            self.data[k][label] = v

    def load_text(self, filename, label="", filetype=None, parser=None):
        """ Register a text file with the library. The label is an optional label you’ll use in your
            visualizations to identify the text
        :param filename: (str) filename of the input file
        :param label: (str) unique label for a text file that we parsed
        :param filetype: (str) filetype of the input file
        :param parser: (str) this is the stop word
        :return: pre-processes, save results, and count the number of loaded files
        """
        assert self.counter < 10, 'You cannot add more files'
        assert filename is not None, 'File name is empty!'
        assert 'txt' in filename or filetype in filename, 'This is not the correct filetype'

        assert isinstance(filename, str), 'You chose the wrong data type'
        assert isinstance(label, str), 'You chose the wrong data type'
        assert filetype is None or isinstance(filetype, str), 'You chose the wrong data type'
        assert parser is None or \
               (isinstance(parser, list) and reduce(lambda a, b: a and b, [isinstance(v, str) for v in parser])), \
            'You chose the wrong data type'

        # preprocesses the file, save the result, and count the number of file loaded
        rslt = self.preprocessor(filename, filetype, parser)
        self._save_results(rslt, label)
        self.counter += 1

    # I felt like this does the parsing job, combined with file-specified openings, according to class nlp example
    @staticmethod
    def load_stop_words(txtfile, stopfile):
        """a function with list input of common or stop words. These get filtered from each file automatically
        :param txtfile: (list of str) list of strings from the input text file
        :param stopfile: (list of str) list of strings to be removed from the txtfile
        :return: (list of str) txtfile with all the stopfile words removed
        """
        assert isinstance(txtfile, list), 'You chose the wrong data type'
        assert reduce(lambda a, b: a and b, [isinstance(v, str) for v in txtfile]), 'You chose the wrong data type'

        # default stopfile value
        if stopfile is None:
            stopfile = nltk.corpus.stopwords.words("english")

        assert isinstance(stopfile, list), 'You chose the wrong data type'
        assert reduce(lambda a, b: a and b, [isinstance(v, str) for v in stopfile]), 'You chose the wrong data type'

        # removal process
        txtfile = [t for t in txtfile if t.lower() not in stopfile]
        return txtfile

    def preprocessor(self, filename, filetype, parser):
        """ Cleans the data, removing unnecessary whitespace, punctuation, and capitalization.
            Gathers some statistics such as word length, readability scores, sentiment, and so on.
        :param filename: (str) filename of the input file
        :param filetype: (str) filetype of the input file
        :param parser: (str) this is the stop word
        :return: (dict) result of pre-processing process (Data Cleansing and Statistics Gathering)
        """
        assert isinstance(filename, str), 'You chose the wrong data type'

        # reading file by each filetype
        f = open(filename, "r")
        if filetype == 'json':
            assert filetype in filename, 'This is not the correct filetype'
            raw = json.load(f)
            assert 'text' in raw, 'This json file is in invalid format.'
            contents = raw['text']
        elif filetype == 'html':
            assert filetype in filename, 'This is not the correct filetype'
            contents = BeautifulSoup(f, 'html.parser').get_text()
        else:
            contents = f.read()
        f.close()

        assert isinstance(contents, str), 'The content of file should be text'
        contents = self.load_stop_words(contents.split(), parser)

        # string to be cut out from each words
        cut = ['“', '"', '-', '—', '[', '{', '(', '<', ',', ':', ';', '!', '?', ']', '}', ')', '>']
        imploded = list()

        # explodes the string, removes according to the cut list, and implodes them back
        for s in contents:
            assert s is not None and s != '', 'This text contains extra empty spaces somehow'
            cutted = [x for x in list(s) if x not in cut]
            if len(cutted) > 0:
                imploded.append(reduce(lambda a, b: a + b, cutted))

        # turns list of word back into sentences
        contents = ' '.join(imploded).lower()

        # statistics result
        results = {
            'wordcount': Counter(contents.split()),
            'numwords': len(contents.split()),
            'tot_word_per_sentence': self.tot_word_per_sentence(contents),
            'tot_syllable_per_word': self.tot_syllable_per_word(contents),
            'ave_word_length': self.ave_word_length(contents),
            'tot_syllable_per_sentence': self.tot_syllable_per_sentence(contents),
            'flesch_reading_ease': self.flesch_ease(contents),
            'flesch_kincaid_grade_level': self.flesch_kincaid(contents),
            'sentiment_polarity': sia.polarity_scores(contents),
            'sentiment_score': sia.polarity_scores(contents)['compound']
        }
        return results

    def flesch_ease(self, txt):
        # Flesch Reading Ease Score = 206.835-1.015*(Total Words/Total Sentences)-84.6*(Total Syllables/Total Words)
        # source: https://readabilityformulas.com/flesch-reading-ease-readability-formula.php
        """ Calculates the Flesch Reading Ease Score
        :param txt: (str) string to be calculated
        :return: (float) calculated flesch reading ease score of txt
        """
        assert txt is not None, 'The text is empty'
        assert isinstance(txt, str), 'You chose the wrong data type'

        # score calculation and range settings
        score = (206.835 - (1.015 * (self.tot_word_per_sentence(txt))) - (84.6 * self.tot_syllable_per_word(txt)))
        if score > 100:
            score = 100
        elif score < 0:
            score = 0

        return score

    def flesch_kincaid(self, txt):
        # Flesch Kincaid Grade Level = 0.39*(Total Words/Total Sentences) + 11.8*(Total Syllables/Total Words) - 15.59
        # source: https://readabilityformulas.com/flesch-grade-level-readability-formula.php
        """ Calculates the Flesch Kincaid Grade Level Score
        :param txt: (str) string to be calculated
        :return: (float) calculated flesch kincaid grade level score of txt
        """
        assert txt is not None, 'The text is empty'
        assert isinstance(txt, str), 'You chose the wrong data type'

        # score calculation and range settings
        score = ((0.39 * self.tot_word_per_sentence(txt)) + (11.8 * self.tot_syllable_per_word(txt)) - 15.59)
        if score > 18:
            score = 18
        elif score < 0:
            score = 0

        return score

    @staticmethod
    def tot_word_per_sentence(txt):
        """ Calculates the average number of word per sentence
        :param txt: (str) string to be calculated
        :return: (float) calculated average number of word per sentence in txt
        """
        assert txt is not None, 'The text is empty'
        assert isinstance(txt, str), 'You chose the wrong data type'

        # split by sentence and word
        sen = txt.split('.')
        words = txt.split()

        # return by total number of words / total number of sentences
        return len(words) / len(sen)

    @staticmethod
    def syllable_count(txt):
        # source: https://stackoverflow.com/questions/46759492/syllable-count-in-python
        """ Counts the number of syllables in string
        :param txt: (str) string to be counted
        :return: (int) counted number of syllables in txt
        """
        assert txt is not None, 'The text is empty'
        assert isinstance(txt, str), 'You chose the wrong data type'

        # split by words and set default values of word, count, and vowels to test
        word = txt.lower()
        count = 0
        vowels = "aeiouy"

        # first char in vowel?
        if word[0] in vowels:
            count += 1

        # mid-word syllable rule applied
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1

        return count

    def tot_syllable_per_word(self, txt):
        """ Calculates the average number of syllable per word
        :param txt: (str) string to be calculated
        :return: (float) calculated number of syllable per word in txt
        """
        assert txt is not None, 'The text is empty'
        assert isinstance(txt, str), 'You chose the wrong data type'

        # split by word, count the number of words, and count the number of syllables
        word_list = txt.split()
        total_syllables = sum([self.syllable_count(x) for x in word_list])
        total_words = len(word_list)

        # return by total number of syllables / total number of words
        return total_syllables / total_words

    @staticmethod
    def ave_word_length(txt):
        """ Calculates the average word length
        :param txt: (str) string to be calculated
        :return: (float) calculated average word length
        """
        assert txt is not None, 'The text is empty'
        assert isinstance(txt, str), 'You chose the wrong data type'

        # list of each word length
        word_list = map(lambda x: len(x), txt.split())

        return mean(word_list)

    def tot_syllable_per_sentence(self, txt):
        """ Calculates the average number of syllable per sentence
        :param txt: (str) string to be calculated
        :return: (float) calculated average number of syllable per sentence
        """
        assert txt is not None, 'The text is empty'
        assert isinstance(txt, str), 'You chose the wrong data type'

        # split by sentences and count the syllables
        sen = txt.split('.')
        syl = self.syllable_count(txt)

        # returns the total number of syllables / total number of sentences
        return syl / len(sen)

    # Text-to-Word Sankey diagram
    def wordcount_sankey(self, word_list=None, k=5):
        """ Map each text to words using a Sankey diagram, where the thickness of the line
            is the number of times that word occurs in the text. Users can specify a particular
            set of words, or the words can be the union of the k most common words across
            each text file (excluding stop words).
        :param word_list: (list of str) list of strings that users can specify
        :param k: (int) integer to specify the number of most common words to be visualized
        :return: visualize the sankey diagram of relationship between files and words
        """
        assert isinstance(k, int), 'You chose the wrong data type'

        # DataFrame column default values
        text = list()
        word = list()
        count = list()

        # word_list default value if no input
        if word_list is None:
            word_list = list(self.total_wordcounts().keys())[:k]

        assert isinstance(word_list, list), 'You chose the wrong data type'
        assert reduce(lambda a, b: a and b, [isinstance(v, str) for v in word_list]), 'You chose the wrong data type'
        assert self.wl_checker(word_list), 'Not all the words in word list are included in dataset'

        # setting DataFrame values from 'wordcount'
        for label, wc in self.data['wordcount'].items():
            for key, val in dict(wc).items():
                if key in word_list:
                    word.append(key)
                    count.append(val)
                    text.append(label)

        # create DataFrame and make Sankey based on the new DataFrame
        df = pd.DataFrame({'text': text, 'word': word, 'count': count})
        sk.make_sankey(df, 'text', 'word', 'count')

    def total_wordcounts(self):
        """ Counts the total word count from defaultdict['wordcount'],
            regardless of label in descending number of appearance
        :return: (dict) the total word count
        """
        assert 'wordcount' in self.data, 'Word Count Stats does not exist'

        # adds number of word appearance, regardless of files to new dictionary
        wordcount = defaultdict(int)
        for label, wc in self.data['wordcount'].items():
            for key, val in dict(wc).items():
                wordcount[key] += val

        # re-orders and preserves the order of dictionary into descending order
        # source: https://www.geeksforgeeks.org/ordereddict-in-python/
        wordcount = OrderedDict({k: v for k, v in sorted(wordcount.items(), key=lambda item: item[1], reverse=True)})

        return wordcount

    def wl_checker(self, wl):
        """ Checks if all the words from the wl are in data
        :param wl: (list of str) list of string to be checked
        :return: (bool) boolean if all the contents from wl are in data
        """
        assert isinstance(wl, list), 'You chose the wrong data type'
        assert reduce(lambda a, b: a and b, [isinstance(v, str) for v in wl]), 'You chose the wrong data type'

        return reduce(lambda a, b: a and b, [w in self.total_wordcounts().keys() for w in wl])

    # subplots
    def top_tens(self, d='wordcount', sort=True):
        """ Map subplots to each label (textfiles) about data d of the user choice.
            Allows the user to choose if this dataset to be sorted in descending order or disorganized.
            The maximum number of outputs for each subplots are 10.
        :param d: (str) data of the user choice to be visualized
        :param sort: (bool) True for descending order data, False for disorganized data
        :return: visualize the subplots of barplot from input data d
        """
        assert d in self.data, 'Your input data is invalid'
        assert reduce(lambda a, b: a and b, [isinstance(v, dict) for v in list(self.data[d].values())]),\
            'You chose the wrong data type'

        # values set
        i = 0
        data1 = self.data[d]

        # subplots setup
        fig, axs = plt.subplots(1, len(data1))
        fig.suptitle(d)

        for label, wc in data1.items():
            # re-orders and preserves the order of dictionary if no user input
            if sort:
                wc = OrderedDict(
                    {k: v for k, v in sorted(wc.items(), key=lambda item: item[1], reverse=True)})

            #  until first 10 inputs, outputs horizontal bar graph of dictionary contents
            t = 0
            for w, c in wc.items():
                assert isinstance(c, int) or isinstance(c, float), 'You chose the wrong data type'
                axs[i].barh(w, c)
                t += 1
                if t == 10:
                    break

            # sets title, and re-orders into actual descending order.
            # (it was top 10s ascending because of for loop system)
            axs[i].set_title(label)
            axs[i].invert_yaxis()
            i += 1
        plt.show()

    # overlays info from each text file onto a single visualization
    def score_comparison(self, d='sentiment_score'):
        """ Map barplot of scores d from each textfiles to compare in one visualization.
            It is legend-ed and shows data in descending order
        :param d: (str) data of the user choice to be visualized
        :return: visualize the barplot of all the files with input data d
        """
        assert d in self.data, 'Your input data is invalid'
        assert reduce(lambda a, b: a and b, [isinstance(v, int) or isinstance(v, float)
                                             for v in list(self.data[d].values())]), 'You chose the wrong data type'

        # re-orders and preserves the order of dictionary  into descending order
        s_dict = OrderedDict(
            {k: v for k, v in sorted(self.data[d].items(), key=lambda item: item[1], reverse=True)})

        # plots the bar graph, legends, and sets title
        for label, score in s_dict.items():
            plt.bar(label, score)
        plt.legend()
        plt.suptitle(d)
        plt.show()


def main():
    # initialize the Txtanalyzer object
    ta = Txtanalyzer()

    # load all the files
    ta.load_text('textfiles/all_i_want_for_christmas_is_you_lyrics.txt', 'A')
    ta.load_text('textfiles/happy_xmas_war_is_over_lyrics.txt', 'HX')
    ta.load_text('textfiles/holly_jolly_christmas_lyrics.txt', 'HJC')
    ta.load_text('textfiles/its_beginning_to_look_a_lot_like_christmas.txt', 'I')
    ta.load_text('textfiles/jingle_bell_rock.txt', 'J')
    ta.load_text('textfiles/last_christmas_lyrics.txt', 'LC')
    ta.load_text('textfiles/let_it_snow_3_lyrics.txt', 'LIS')
    ta.load_text('textfiles/rockin_around_the_christmas_tree.txt', 'R')
    ta.load_text('textfiles/sleigh_ride_lyrics.txt', 'S')
    ta.load_text('textfiles/wonderful_christmas_time_lyrics.txt', 'W')

    # print test statistics
    pp.pprint(ta.data)

    # visualizations
    ta.wordcount_sankey()
    ta.top_tens()
    ta.top_tens('sentiment_polarity', sort=False)
    ta.score_comparison()
    ta.score_comparison('flesch_kincaid_grade_level')
    ta.score_comparison('sentiment_score')


if __name__ == '__main__':
    main()
