import re

import nltk
# nltk.download('wordnet')
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

stemmer = SnowballStemmer(language='english')
lemmatizer = WordNetLemmatizer()

sentence_pattern_split = r'(?<=\.|!|\?)(?<![A-Z]{1}\.)(\s)'
word_pattern = r'#TOKEN#|[A-Za-z]+[\'\`][a-z]|\w+' + '|' + r'(?<![A-Z])\.\.\.|\.|,|!|\?|:|;'

token_specification = {
    'DATE':     r'\b(\d{1,2})[.\-/](\d{2})[.\-/](\d{2,4})\b|' + \
                    r'\b(\d{2})\s([A-Z][a-z]+)\s(\d{4})\b',
    'PHONE':    r'(?:\+7[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{4})|'    +\
                    r'(?:8[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4})',
    'EMAIL':    r'(:?[a-zA-Z0-9_.+-]+@[a-pr-zA-PRZ0-9-]+\.[!\w\d+-]+)',
    'TIME':     r'\b(0?[1-9]|1[0-9]|2[0-3]):([0-5][0-9])(:([0-5][0-9]))?\b|(\b(0?[1-9]|1[0-2]):([0-][0-9])\s([APap][Mm])\b)', 
    'NAMINGS':  r'\b([A-Z][a-z]{2,})\s([A-Z][a-z]+)|'      +\
                    r'\b([A-Z])\.\s([A-Z][a-z]+)\b|'        
                    r'(?<!\.\s)(\b[A-Z][a-z]*\b)',
}
common_pattern = '|'.join(token_specification.values())

def get_token_annotation(token):
    return '\t'.join((token, stemmer.stem(token), lemmatizer.lemmatize(token))) + '\n'

def get_special_token_annotation(token):
    return '\t'.join([token for _ in range(3)]) +'\n'

def tokenize_text(text):
    text_annotation = ''

    sentences = re.split(sentence_pattern_split, text)
    for sentence in sentences:
        sentence_annotation = ''

        # достаём специальные токены - наши усложнённые ситуации
        matches = list(re.finditer(common_pattern, sentence))

        # исключаем невалидные совпадения, валидные сохраняем в порядке возникновения
        special_tokens = []
        for m in matches:
            if len(m.group(0)) > 0:
                special_tokens.append(m.group(0))
                sentence = sentence.replace(m.group(0), '#TOKEN#')

        # находим все слова
        tokens = re.findall(word_pattern, sentence)

        for token in tokens:
            if token == '#TOKEN#':
                token = special_tokens.pop(0)
                sentence_annotation += get_special_token_annotation(token)
            else:
                sentence_annotation += get_token_annotation(token)

        
        text_annotation += sentence_annotation + '\n'

    return text_annotation, tokens