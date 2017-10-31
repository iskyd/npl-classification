Setting up virtual env
===================

    virtualenv my_app
    source my_app/bin/activate

Get npl-classification
===================

    cd my_app
    git clone https://github.com/iskyd/npl-classification ./

Install requirements
===================

    bin/pip install -r requirements.txt
    bin/python -m nltk.downloader -d /path/to/my_app stopwords punkt

Get with Docker
===================

    docker pull iskyd/npl-classification
    docker run -t -d iskyd/npl-classification
    docker exec -ti <container hash> bash

To copy your data from localhost to container you can use docker cp

    docker cp data.json <container hash>:/app/data.json

Read more info about [Docker](https://docs.docker.com/)

Usage
===================
    python npl-classification/classify.py --help

**Create train and dump classifier**

    python npl-classification/classify.py --create-classifier='/path/to/document.json' --dump-classifier='/path/to/dump.picke'

Your document.json is just a simply json document with the text and the category

    {
        text: 'Test message',
        category: 'Test category'
    }

You can also have a different structure and specify the fields of text and category adding

    --text-field='mycustomfield'
    --classification-field='mycustomclassificationfield

Default language for stemmer stopwords and word tokenizer is english, you can specify a different language using

    --stemmer-language='italian'
    --stopwords-language='italian'
    --word-tokenize-language='italian'

If you want to ignore stopwords in stemmer just specify -i options.

By default stopwords is removed from training_set, if you want to keep it for training/accuracy/classify just add -k option

You can specify the number of row used for training using the --row-training-set options by default is 1000
    --row-training-set=100

Using the -r options the row from document.json is read random (Default is False).

**Load classifier**

    python npl-classification/classify.py --load-classifier='/path/to/dump.picke'

**Testing classifier**
During the test you can specify the creation/training of classifier or the load of classifier in the same way we've seen before.
The option -a is used to calculate the accuracy and perform test from a json file.
You can specify the json file using --test-file-path.
In the same way we seen before you can specify the number of row used, get it random or no, specify the text field and the classification field of json file and set the language for stemmer, stopwords and word-tokenize.

    python npl-classification/classify.py --load-classifier='/path/to/dump.picke' -a --test-file-path='/path/to/data.json' --random-row-test-set

You can use some other options:

    --test-text-field
    --test-classification-field
    --row-test-set (in order to set limit to the number of row you want to use for test, default=500)

**Classify**
As seen before you can create or load the classifier.

    python npl-classification/classify.py --load-classifier='/path/to/dump.picke' --classify='I want classify this text'

**More info**
To get more info use --help command.
