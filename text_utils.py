import re
import numpy as np
from nltk.corpus import stopwords


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
    l_stopwords = stopwords.words(lang)
    content = [w for w in l_words if w.lower() not in l_stopwords]
    return content


def binarizeText(text, remove_stopwords=False, remove_html=True):
    if remove_html:
        text = removeHtml(text)
    if remove_stopwords:
        text = removeStopwords(text.split(' '))
    text = np.array([ord(c) if ord(c) < 126 else 127 for c in text], dtype=int)
    return np.eye(128, dtype=int)[text]
