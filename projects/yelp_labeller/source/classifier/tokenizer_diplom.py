import re
import string

import fitz
import os
import docx2txt
import logging

import inflect
import nltk
from nltk import LancasterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
stopwords = set(stopwords.words('english'))


class Convertor:
    def __init__(self):
        self.extension = {'.docx': self.from_doc, '.pdf': self.from_pdf, '.txt': self.from_txt}
        self.regex_dict = {r"Reviewer [0-9] Report\W": "",  # delete Reviewer text
                           r"Author Response(\s|.)*": "",  # delete Author response text (link to the file)
                           # r" +": " ",  # matches any number of spaces > space
                           r"\W+": " ",  # matches any non-word character > space
                           r"\u00A0+": " "  # matches   (no-break-space) > space
                           }
        self.pipeline = [self.remove_regex,
                         self.remove_unprintable,
                         self.text_lower,
                         self.number_to_text,
                         self.tokenize,
                         self.stopword,
                         # self.stemming,
                         self.lemmatization]

    def save_to_txt(self, text, path_to, file_name):
        fill_path_to = os.path.join(path_to, file_name + ".txt")
        with open(fill_path_to, 'w', encoding="UTF-8") as file:
            file.writelines(text)

    def from_pdf(self, path):
        """
        - path: path to file
        """
        doc = fitz.open(path)
        text = [page.get_text() for page in doc]
        return text

    def from_doc(self, path):
        """
        - path: path to file
        """
        return docx2txt.process(path)

    def from_txt(self, path):
        with open(path, 'r', encoding="UTF-8") as file:
            text = file.readlines()

        return '\n'.join(text)

    def convert_from_dir(self, path, path_to, article_name, annotation_path=None):
        """
        - path: path to file
        - path_to: directory to save
        - file_name: filename without extension
        """
        text = []
        for annotation in os.listdir(path):
            file_ext = os.path.splitext(annotation)[-1]
            fill_path = os.path.join(path, annotation)
            if file_ext not in self.extension:
                continue
            if annotation[:2] == "~$":
                continue
            new_text = self.extension[file_ext](fill_path)
            new_text = self.apply_pipeline(new_text)
            if new_text == '' or new_text == []:
                continue
            text.extend(new_text)
        self.save_to_txt(" ".join(text), path_to, article_name)
        if annotation_path is not None:
            self.add_annotation(text, annotation_path, article_name, path_to)

    def apply_pipeline(self, text) -> list[str]:
        for func in self.pipeline:
            text = func(text)
        return text

    def remove_unprintable(self, text):
        return text
        # Добавляем апостроф &#8217
        # my_printable = string.printable+chr(8217)
        # return "".join(filter(lambda x: x in my_printable, text))

    def text_lower(self, text):
        return text.lower()

    def number_to_text(self, data):
        temp_str = data.split()
        string = []
        engine = inflect.engine()
        for i in temp_str:
            # if the word is digit, converted to
            # word else the sequence continues
            if i.isdigit():
                temp = engine.number_to_words(i)
                string.append(temp)
            else:
                string.append(i)
        return " ".join(string)

    def tokenize(self, text) -> list[str]:
        return nltk.word_tokenize(text)

    def stopword(self, text_data):
        clean = []
        for i in text_data:
            if i not in stopwords:
                clean.append(i)
        return clean

    def stemming(self, text_data):
        # Обрезает слова
        stemmer = LancasterStemmer()
        stemmed = []
        for i in text_data:
            stem = stemmer.stem(i)
            stemmed.append(stem)
        return stemmed

    def lemmatization(self, text_data):
        # Приводит слова в начальную форму
        lemma = WordNetLemmatizer()
        lemmas = []
        for i in text_data:
            lem = lemma.lemmatize(i, pos='v')
            lemmas.append(lem)
        return lemmas

    def remove_regex(self, text):
        for regex, repl in self.regex_dict.items():
            text = re.sub(regex, repl, text)
        return text

# def multi_split(text, separators):
#     arr = [text]
#     split_regex = r"(.* ?({separators}) )".format(separators="|".join(map(lambda x: fr"\{x}", separators)))
#     for separator in separators:
#         for _ in range(len(arr)):
#             sentence = arr.pop(0)
#             arr.extend(filter(lambda x: x!="", re.split(fr"(.*\{separator} )", sentence)))
#             # arr.extend([s+separator if idx%2==0 else s for idx, s in enumerate(sentence.split(separator)) if s != ""])
#     return arr

def multi_split(text, separators):
    split_regex = r"(.*?({separators}) )".format(separators="|".join(map(lambda x: fr"\{x}", separators)))
    arr = re.split(split_regex, text)
    arr = list(filter(lambda x: x != '' and x not in separators, arr))
    return arr


if __name__ == '__main__':
    # multi_split("1234. 4342? 111", ".?")
    # multi_split("Why Jung? The authors have selected Jung’s archetypes as the theoretical foundation for the paper.",
    #             ".?")
    print_annotated_text("./data_test/admsci6020005.ann", per_one_row=25)
    # converter = Convertor()
    # converter.convert_from_dir(
    #     r"../ReviewArgumentationFramework/mdpi_review/reviewed_articles\admsci6020005\sub-articles",
    #     "./data_test/",
    #     "admsci6020005",
    #     annotation_path="../ReviewArgumentationFramework/mdpi_annotated/admsci6020005_makarova")
