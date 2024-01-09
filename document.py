"""
Cynthia Hong
CSE 163 AF
This file is for representing a single document in
the SearchEngine. It includes three methods, which are
term_frequency, get_path, and get_words.
"""

from cse163_utils import normalize_token


class Document:
    """
    class Document represents the data in a single web
    page and includes methods to compute term frequency.
    """
    def __init__(self, name):
        """
        Takes a path to a document and initializes the document
        data. Assume that the file exists, but that it could be empty.
        In order to implement the method later.
        """
        self._name = name
        self._count_term = {}
        count = {}
        word_count = 0
        with open(name) as file_name:
            for line in file_name.readlines():
                words = line.split()
                for word in words:
                    word_count += 1
                    term = normalize_token(word)
                    if term in count:
                        count[term] += 1
                    else:
                        count[term] = 1
        for term, number in count.items():
            self._count_term[term] = number / word_count

    def term_frequency(self, term):
        """
        Takes a term.
        Returns the term frequency of a given term by
        looking it up in the precomputed dictionary.
        Normalizes the term.
        If a term does not appear in a given
        document, returns a term frequency of 0.
        """
        term = normalize_token(term)
        if term not in self._count_term.keys():
            return 0
        else:
            return self._count_term[term]

    def get_path(self):
        """
        Returns the path of the file that this
        document represents.
        """
        return self._name

    def get_words(self):
        """
        Returns a list of the unique, normalized
        words in this document.
        """
        return list(self._count_term.keys())
