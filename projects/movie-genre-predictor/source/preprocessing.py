import datetime

import pandas as pd
from pathlib import Path
import os
import re
from nltk import WordNetLemmatizer, SnowballStemmer
import nltk

nltk.download('wordnet')


def split_to_words(sentence):
    # should be somewhere else, here just to avoid scrolling
    raw = re.sub(r'\b(MD|PhD|Mr|Jr|M\.?D\.?|P\.?h\.?D\.?|Mister|Junior)\s+(\w+)\b', "Title", raw)
    raw = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "email", raw)
    raw = re.sub(r'^\d{2}[ ](January|February|March|April|May|June|July|August|September|October|November|December)[ ]\d{2,4}', "date", raw)
    raw = re.sub(r'^\d{2,4}[-/.]\d{2,4}[-/.]\d{2,4}', "date", raw)


    words = re.findall(r"(((?<=^)|(?<= )|(?<=\"))(\w+[-,.@]?)+\w*\.?)", sentence)
    words = list(map(lambda x: x[0], words))
    return words


def split_to_sentences(lines):
    
    raw = "".join(lines)

    # remove non-ascii characters
    raw = re.sub(r'[^\x00-\x7f]', '', raw)

    # remove extra whitespace. Could've used \s
    raw = re.sub(r" {2,}|\t+", " ", raw)  
    raw = re.sub(r"^ ", "", raw)
    raw = re.sub("\n{2,}", "\n", raw)  
    raw = re.sub("\n ", "\n", raw)  

    # split into sentences
    # sentences = re.split(r"[\.,!?] ", raw)  # easier but way dumber
    sentences = re.split(
        r"(((?<!\w\.\w.)(?<!\s\w\.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s(?=[A-Z]))|((?<![\,\-\:])\n(?=[A-Z]|\" )))", raw)[::4]
    
    # todo recover punctuation, registers (upper-lower), individual tokens 
    # at least for first column 

    return sentences


def process_file(filepath, result_filepath):
    wnl = WordNetLemmatizer()
    sst = SnowballStemmer("english")
    with open(filepath) as f:
        lines = f.readlines()
        sentences = split_to_sentences(lines)
        all_words = []
        for s in sentences:
            all_words += split_to_words(s)
            all_words.append("\n")
        lemmatized = []
        stemmed = []
        original = []
        for w in all_words:
            w_processed = re.sub(r"[.!?,]$", "", w).lower()  # remove punctuation
            lemmatized.append(wnl.lemmatize(w_processed))
            stemmed.append(sst.stem(w_processed))
            original.append(w_processed)

    with open(result_filepath, "w") as f:
        for i in range(len(original)):
            if original[i] == "\n":
                print("", file=f)
            else:
                print(original[i], stemmed[i], lemmatized[i], sep="\t", file=f)


def process_topic(dirname, result_dirname, topic):
    topic_dir = os.path.join(dirname, topic)
    result_dir = os.path.join(result_dirname, topic)

    if not os.path.isdir(result_dir):
        Path(result_dir).mkdir(exist_ok=True, parents=True)

    files = os.listdir(topic_dir)
    for f in files:
        path = os.path.join(topic_dir, f)
        result_path = os.path.join(result_dir, f + ".tsv")
        process_file(path, result_path)


if __name__ == "__main__":
    df = pd.read_csv("../assets/raw_data/movie_meta_data.csv", index_col='imdbid')
    df[['PrimaryGenre', 'SecondaryGenres']] = df['genres'].str.split(', ', n=1, expand=True)

    data_dir = os.path.realpath("../assets/raw_data/raw_texts")
    assets_dir = os.path.realpath("../assets/annotated_corpus")

    train_dir = os.path.join(data_dir, "train")
    result_train_dir = os.path.join(assets_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    result_test_dir = os.path.join(assets_dir, "test")

    topics = set()
    for genre in df['PrimaryGenre']:
        topics.add(genre)

    from sklearn.model_selection import train_test_split

    train, test = train_test_split(df)

    for genre in topics:
        Path(f'{result_train_dir}/{genre}', ).mkdir(exist_ok=True, parents=True)
        Path(f'{result_test_dir}/{genre}', ).mkdir(exist_ok=True, parents=True)

    errors = 0
    movies = 0
    corrupted = 0
    for file in os.listdir(data_dir):
        try:
            imdbid = (file[file.rfind('_') + 1:file.rfind('.')])
            # print(df.loc[int(imdbid)]['PrimaryGenre'])
            movies += 1
        except KeyError:
            errors += 1
        except ValueError:
            print(f"File corrupted: {file[file.rfind('_') + 1:file.rfind('.')]}")
    print(f"Movies found {movies}")
    print(f"Movies not found {errors}")
    print(f"Files corrupted {corrupted}")
    print(f'Total genres: {len(topics)}')
    start = datetime.datetime.now().timestamp()
    other_errors = 0
    decode_errors = 0

    for file in os.listdir(data_dir):
        imdbid = (file[file.rfind('_') + 1:file.rfind('.')])
        try:
            if train.loc[int(imdbid)].get('PrimaryGenre', False):
                process_file(f'{data_dir}/{file}',
                             f'{result_train_dir}/{train.loc[int(imdbid)].get("PrimaryGenre")}/{imdbid}.tsv')
        except KeyError:
            pass
        except UnicodeDecodeError:
            decode_errors += 1
        except Exception as e:
            print(e)
            other_errors += 1
        try:
            if test.loc[int(imdbid)].get('PrimaryGenre', False):
                process_file(f'{data_dir}/{file}',
                             f'{result_test_dir}/{test.loc[int(imdbid)].get("PrimaryGenre")}/{imdbid}.tsv')
        except KeyError:
            pass
        except UnicodeDecodeError:
            decode_errors += 1
        except Exception as e:
            print(e)
            other_errors += 1

    print(f'Decode errors: {decode_errors}')
    print(f'Other errors: {other_errors}')
