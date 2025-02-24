import unittest
from projects.yelp_labeller.source.classifier.tokenizer import tokenize

class TestTokenization(unittest.TestCase):
    def test_on_empty_string(self):
        self.assertEqual(tokenize(""), [])

    def test_simple(self):
        email_1 = "Hello word, Jonny! 1234"
        self.assertEqual([["Hello","word",",", "Jonny", "!"], ["1234"]], tokenize(email_1))

    def test_on_spaces(self):
        self.assertEqual([["foo", "bar"]],tokenize("foo bar"))

    def test_on_empty_string_with_spaces(self):
        self.assertEqual([[]],tokenize("    "))

    def test_on_tabs_and_newlines(self):
        self.assertEqual([["foo", "bar", "baz", "qux", "quux"]],tokenize("foo\tbar\nbaz    qux\t\t\t\nquux"))

    def test_abbr(self):
        self.assertEqual([["dr.", "goldberg", "ms.", "pover", "mss.", "angely"]],
                         tokenize("dr. goldberg ms. pover mss. angely"))
    def test_email(self):
        email_1 = "abc@abc.abc"
        self.assertEqual([[email_1]], tokenize(email_1))
        email_2 = "ab_c.d@gmail.com"
        self.assertEqual( [[email_2]], tokenize(email_2))

    def test_phone_number(self):
        number = "8(921)3215675"
        self.assertEqual([[number]], tokenize(number) )
        number = "8 (921) 3215675"
        self.assertEqual([[number]], tokenize(number) )
        number = "+7 (921) 321 56 75"
        self.assertEqual([[number]], tokenize(number) )
        number = "+7(921)321-56-75"
        self.assertEqual([[number]], tokenize(number) )
        number = "+7 (921) 321-56-75"
        self.assertEqual([[number]], tokenize(number) )
        number = "+7 (921) 321-5675"
        self.assertEqual([[number]], tokenize(number) )
        number = "321-56-75"
        self.assertEqual([[number]], tokenize(number) )

    def test_prices(self):
        price = "$800"
        self.assertEqual([[price]], tokenize(price))
        price = "$800.00"
        self.assertEqual([[price]], tokenize(price))
        price = "$ 800.00"
        self.assertEqual([[price]], tokenize(price))
        price = "800$"
        self.assertEqual([[price]], tokenize(price))
        price = "800.00$"
        self.assertEqual([[price]], tokenize(price))
        price = "800.00 $"
        self.assertEqual([[price]], tokenize(price))
        price = "1$"
        self.assertEqual([[price]], tokenize(price))
        price = "$8.09"
        self.assertEqual([[price]], tokenize(price))
        price = "8.12 $"
        self.assertEqual([[price]], tokenize(price))


if __name__ == "__main__":
    unittest.main()
