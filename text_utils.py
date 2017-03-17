import re
import numpy as np
import pdb


def removeHtml(html):
    """
    Copied from NLTK package.
    Remove HTML markup from the given string.

    :param html: the HTML string to be cleaned
    :type html: str
    :rtype: str
    """

    # First we remove inline JavaScript/CSS:
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
    # Then we remove html comments. This has to be done before removing regular
    # tags since comments can contain '>' characters.
    cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
    # Next we can remove the remaining tags:
    cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
    # Finally, we deal with whitespace
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    cleaned = re.sub(r"  ", " ", cleaned)
    return cleaned.strip()


def removeStopwords(l_words, lang='english'):
    from nltk.corpus import stopwords
    l_stopwords = stopwords.words(lang)
    content = [w for w in l_words if w.lower() not in l_stopwords]
    return content


def binarizeText(text, encoder, lower_case=True, remove_stopwords=False,
                 remove_html=True):
    from bs4 import BeautifulSoup
    if remove_html:
        try:
            text = BeautifulSoup(text, 'lxml').get_text()
        except:
            pdb.set_trace()
    if lower_case:
        preprocess = lambda x: x.lower()
    else:
        preprocess = lambda x: x

    if remove_stopwords:
        text = removeStopwords(text.split(' '))
    text = np.array(encoder.encode(preprocess(text)), dtype=int)
    return np.eye(len(encoder.alphabet), dtype=int)[:, text]


class textEncoder():
    def __init__(self, alphabet, out='#'):
        self.alphabet = alphabet
        self.out = '#'
        self.encoder = {}
        self.decoder = {}
        for i in range(len(alphabet)):
            self.encoder[alphabet[i]] = i
            self.decoder[i] = alphabet[i]

    def encode(self, data):
        return [self.encoder[x]
                if x in self.encoder else len(self.encoder) + 1
                for x in data]

    def decode(self, data):
        return [self.decoder[x]
                if x in self.decoder else self.out
                for x in data]
