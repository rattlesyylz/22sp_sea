"""
Cynthia Hong
CSE 163 AF
This file is the file representing the SearchEngine class
as well as a command-line interface for using the SearchEngine.
It includes two methods, which are
_calculate_idf and get_path, and get_words.
"""

import os
import math
from document import Document
from cse163_utils import normalize_token
from operator import itemgetter


class SearchEngine:
    """
    The SearchEngine class defined in search_engine.py
    represents a corpus of Document objects and includes
    methods to compute the tf–idf statistic between each
    document and a given query.
    """
    def __init__(self, name):
        """
        Takes a str path to a directory such as /course/small_wiki/
        and constructs an inverted index associating each term in the
        corpus to the list of documents that contain the term.
        """
        self._name = os.listdir(name)
        self._dic = {}
        docs = []
        for file_name in self._name:
            docs.append(file_name)
        for doc in docs:
            document = Document(os.path.join(name, doc))
            for word in document.get_words():
                if word in self._dic.keys():
                    self._dic[word].append(document)
                else:
                    self._dic[word] = []
                    self._dic[word].append(document)

    def _calculate_idf(self, voc):
        """
        Takes a str term.
        Returns the inverse document frequency of that term.
        If the term is not in the corpus, return 0.
        Otherwise, if the term is in the corpus, compute the
        inverse document frequency idf as follows.
        """
        word = normalize_token(voc)
        if word not in self._dic.keys():
            return 0
        else:
            return math.log(len(self._name) / len(self._dic[word]))

    def search(self, query):
        """
        Takes a str query that contains one or more terms.
        Returns a list of document paths sorted in descending
        order by tf–idf statistic.
        Normalizes the terms before processing.
        If there are no matching documents, return an empty list.
        """
        docs = set()
        get_answer = []
        get_result = []
        final_result = []
        terms = query.split()
        for term in terms:
            token = normalize_token(term)
            if token in self._dic.keys():
                for doc in self._dic[token]:
                    docs.add(doc)

        for document in docs:
            tfidf = 0
            for term in terms:
                tfidf += self._calculate_idf(term) * \
                            document.term_frequency(term)
            get_answer.append((document.get_path(), tfidf))

        if len(get_answer) == 0 or len(docs) == 0:
            return get_answer
        get_result = sorted(get_answer, key=itemgetter(1), reverse=True)

        for temp in get_result:
            final_result.append(temp[0])
        return final_result
