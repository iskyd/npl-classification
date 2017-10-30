from optparse import OptionParser
import sys
from classification import Classification

def main():
    parser = OptionParser()
    parser.add_option("--stemmer-language", dest="stemmer_language", help="Language for SnowballStemmer", default="english")
    parser.add_option('-i', action="store_true", dest="ignore_stopwords_stemmer", help="Ignore stopwords in stemmer, default false", default=False)
    parser.add_option("--stopwords-language", dest="stopwords_language", help="Language for stopwords")
    parser.add_option("-k", action="store_true", dest="keep_stopwords", help="Keep stopwords, default remove", default=False)
    parser.add_option('--load-classifier', dest="load_classifier_file_path", help="Specify load classifiers file")
    parser.add_option('--create-classifier', dest="create_classifier", help="File for training set")
    parser.add_option('--row-training-set', dest="row_training_set", help="Number of row for training set", default=1000)
    parser.add_option('-r', action="store_true", dest="random_row_training_set", help="Get random row from training set file", default=False)
    parser.add_option('--text-field', dest="text_field", help="text field in json file", default="text")
    parser.add_option('--word-tokenize-language', dest="word_tokenize_language", help="Word tokenize language", default="english")
    parser.add_option('--classification-field', dest="classification_field", help="Classification field in json data", default="category")
    parser.add_option('--dump-classifier', dest="dump_classifier", help="Dump classifier file", default=False)
    parser.add_option('-a', action="store_true", dest="calculate_accuracy", help="Calculate accuracy", default=False)
    parser.add_option('--test-file-path', dest="test_file_path", help="Test file path")
    parser.add_option('--row-test-set', dest="row_test_set", help="Number of row for test set", default=500)
    parser.add_option('--random-row-test-set', action="store_true", dest="random_row_test_set", help="Get random row from test set file", default=False)
    parser.add_option('--test-text-field', dest="test_text_field", help="text field in json test file", default="text")
    parser.add_option('--test-classification-field', dest="test_classification_field", help="classificaion field in json test file", default="category")
    parser.add_option('--classify', dest="classify_text", help="classify text", default=False)
    (options, args) = parser.parse_args(sys.argv)

    cl = Classification(
        stemmer_language=options.stemmer_language,
        stopwords_language=options.stopwords_language,
        ignore_stopwords_stemmer=options.ignore_stopwords_stemmer,
    )

    if options.load_classifier_file_path:
        cl.load_classifier(load_classifier_file_path=options.load_classifier_file_path)
    elif options.create_classifier:
        cl.create_and_train_classifier(
            training_file_path=options.create_classifier,
            keep_stopwords=options.keep_stopwords,
            row_training_set=options.row_training_set,
            random_row_training_set=options.random_row_training_set,
            text_field=options.text_field,
            word_tokenize_language=options.word_tokenize_language,
            classification_field=options.classification_field
        )

    if options.dump_classifier:
        cl.dump_classifier(options.dump_classifier)
    if options.calculate_accuracy:
        cl.accuracy(
            test_file_path=options.test_file_path,
            keep_stopwords=options.keep_stopwords,
            row_test_set=options.row_test_set,
            random_row_test_set=options.random_row_test_set,
            text_field=options.test_text_field,
            word_tokenize_language=options.word_tokenize_language,
            classification_field=options.test_classification_field
        )

    if(options.classify_text):
        cl.classify(
            text=options.classify_text,
            keep_stopwords=options.keep_stopwords,
            word_tokenize_language=options.word_tokenize_language
        )

if __name__ == '__main__':
    main()
