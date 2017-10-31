from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from textblob.classifiers import NaiveBayesClassifier
import pickle
import json
from random import randint

class Classification:
    def __init__(self, stemmer_language, stopwords_language, ignore_stopwords_stemmer):
        self.__snowballStemmer = SnowballStemmer(stemmer_language, ignore_stopwords=ignore_stopwords_stemmer)
        self.__stopWords = set(stopwords.words(stopwords_language))

    def __get_classifier_set_file(self, file_path, keep_stopwords, row_set, random_row_set, text_field, word_tokenize_language, classification_field):
        training_set = []
        try:
            with open(file_path) as data_file:
                json_data = json.load(data_file)
            for x in range(row_set):
                if random_row_set:
                    row = json_data[randint(0, len(json_data) - 1)]
                else:
                    row = json_data[x]

                stemmed_words = []
                words = word_tokenize(row[text_field], language=word_tokenize_language)
                if keep_stopwords:
                    stemmed_words = [self.__snowballStemmer.stem(w) for w in words]
                else:
                    stemmed_words = [self.__snowballStemmer.stem(w) for w in words if not w in self.__stopWords]
                training_set.append((' '.join(stemmed_words), row[classification_field]))

            return training_set
        except Exception as e:
            print('[-] Erorr ' + str(e))

            return False

    def load_classifier(self, load_classifier_file_path):
        try:
            print('[+] Loading Classifier from ' + load_classifier_file_path)
            f = open(load_classifier_file_path, 'rb')
            self.__classifier = pickle.load(f)
            f.close()
        except Exception as e:
            print('[-] Error ' + str(e))

            return False

    def create_and_train_classifier(self, training_file_path, keep_stopwords, row_training_set, random_row_training_set, text_field, word_tokenize_language, classification_field):
        training_set = self.__get_classifier_set_file(
            file_path=training_file_path,
            keep_stopwords=keep_stopwords,
            row_set=row_training_set,
            random_row_set=random_row_training_set,
            text_field=text_field,
            word_tokenize_language=word_tokenize_language,
            classification_field=classification_field
        )
        if training_set:
            print('[+] Creating and training classifier')
            self.__classifier = NaiveBayesClassifier(training_set)
        else:
            print('[-] Cannot create and train classifier')

    def dump_classifier(self, dump_classifier_file):
        try:
            print('[+] Dumping classifier')
            if(dump_classifier_file[-6:] != '.picke'):
                dump_classifier_file = dump_classifier_file + '.picke'
            f = open(dump_classifier_file, 'wb')
            pickle.dump(self.__classifier, f)
            f.close()
        except Exception as e:
            print('[-] Error ' + str(e))

    def accuracy(self, test_file_path, keep_stopwords, row_test_set, random_row_test_set, text_field, word_tokenize_language, classification_field):
        try:
            test_set = self.__get_classifier_set_file(
                file_path=test_file_path,
                keep_stopwords=keep_stopwords,
                row_set=row_test_set,
                random_row_set=random_row_test_set,
                text_field=text_field,
                word_tokenize_language=word_tokenize_language,
                classification_field=classification_field
            )
            print('[+] Calculating accuracy')
            print("Accuracy: {0}".format(self.__classifier.accuracy(test_set)))
        except Exception as e:
            print('[-] Error ' + str(e))

    def classify(self, text, keep_stopwords, word_tokenize_language):
        print('[+] Classifing text: ' + text)
        try:
            stemmed_words = []
            words = word_tokenize(text, language=word_tokenize_language)
            if keep_stopwords:
                stemmed_words = [self.__snowballStemmer.stem(w) for w in words]
            else:
                stemmed_words = [self.__snowballStemmer.stem(w) for w in words if not w in self.__stopWords]
            print(self.__classifier.classify(stemmed_words))
        except Exception as e:
            print('[-] Error ' + str(e))
