import itertools
import re

pattern = r"{price_pattern}|{abbr_patterns}|({phone_pattern})|({email_pattern})|(\'?[\w\-]+)|([^A-Za-z0-9 \n])"
sent_pattern = r"((?<=\.|\?|!|\;))({abbr_patterns})\s"

phone_pattern = r"\+?[0-9] ?\(?[0-9]+\)?[0-9 -]+"
# [\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}
email_pattern = r"[^@ \t\r\n]+@[^@ \t\r\n]+\.[^@ \t\r\n]+"
price_pattern = r"(\$ ?\d*\.?\d+)|(\d*\.?\d+ ?\$)"

english_abbr = ["Mr.", "Mrs.", "Mss.", "Ms.", "Dr."]
english_abbr = [x.replace(".", "\.") for x in english_abbr]
english_abbr.extend(map(lambda x: x.lower(), english_abbr.copy()))

sent_pattern = sent_pattern.format(abbr_patterns="".join(map(lambda x: fr"(?<!{x})", english_abbr)))
sent_pattern = re.compile(sent_pattern)

pattern = pattern.format(abbr_patterns="(" + "|".join(english_abbr) + ")",
                         phone_pattern=phone_pattern,
                         email_pattern=email_pattern,
                         price_pattern=price_pattern)

word_pattern = re.compile(pattern)
# print(word_pattern.pattern.replace(r"//", r"/"))

def split_to_sentence(text: str) -> list[str]:
    return list(filter(lambda x: len(x) if x else False, sent_pattern.split(text)))


def tokenize(text: str) -> list[list[str]]:
    tokenized_sentences = []
    split_sentences = list(filter(lambda x: len(x) if x else False, sent_pattern.split(text)))
    # print(sent_pattern.pattern.replace(r"\\", "\\"))
    for sent in split_sentences:
        # Удаляем экранированные символы
        sent = re.sub(r"\\.", "", sent)
        sent = re.sub(r"(=\s+)", "", sent)
        # Удаляем пробелы
        sent = re.sub(r"\s+", " ", sent)
        words = list(filter(lambda x: x and len(x), itertools.chain(*word_pattern.findall(sent))))
        tokenized_sentences.append(words)

    return tokenized_sentences


if __name__ == "__main__":
    text = "dr. goldberg offers everything i look for in a general practitioner. he's nice and easy to talk to without being patronizing; he's always on time in seeing his patients; he's affiliated with a top-notch hospital (nyu) which my parents have explained to me is very important in case something happens and you need surgery; and you can get referrals to see specialists without having to see him first. really, what more do you need? i'm sitting here trying to think of any complaints i have about him, but i'm really drawing a blank."
    print(tokenize(text))
