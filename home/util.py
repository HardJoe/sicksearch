import math
import re

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


class IdMap:
    """
    Ingat kembali di kuliah, bahwa secara praktis, sebuah dokumen dan
    sebuah term akan direpresentasikan sebagai sebuah integer. Oleh
    karena itu, kita perlu maintain mapping antara string term (atau
    dokumen) ke integer yang bersesuaian, dan sebaliknya. Kelas IdMap ini
    akan melakukan hal tersebut.
    """

    def __init__(self):
        """
        Mapping dari string (term atau nama dokumen) ke id disimpan dalam
        python's dictionary; cukup efisien. Mapping sebaliknya disimpan dalam
        python's list.

        contoh:
            str_to_id["halo"] ---> 8
            str_to_id["/collection/dir0/gamma.txt"] ---> 54

            id_to_str[8] ---> "halo"
            id_to_str[54] ---> "/collection/dir0/gamma.txt"
        """
        self.str_to_id = {}
        self.id_to_str = []

    def __len__(self):
        """Mengembalikan banyaknya term (atau dokumen) yang disimpan di IdMap."""
        return len(self.id_to_str)

    def __get_str(self, i):
        """Mengembalikan string yang terasosiasi dengan index i."""
        # TODO
        return self.id_to_str[i]

    def __get_id(self, s):
        """
        Mengembalikan integer id i yang berkorespondensi dengan sebuah string s.
        Jika s tidak ada pada IdMap, lalu assign sebuah integer id baru dan kembalikan
        integer id baru tersebut.
        """
        # TODO
        try:
            return self.str_to_id[s]
        except KeyError:
            new_id = len(self)
            self.str_to_id[s] = new_id
            self.id_to_str.append(s)
            return new_id

    def __getitem__(self, key):
        """
        __getitem__(...) adalah special method di Python, yang mengizinkan sebuah
        collection class (seperti IdMap ini) mempunyai mekanisme akses atau
        modifikasi elemen dengan syntax [..] seperti pada list dan dictionary di Python.

        Silakan search informasi ini di Web search engine favorit Anda. Saya mendapatkan
        link berikut:

        https://stackoverflow.com/questions/43627405/understanding-getitem-method

        Jika key adalah integer, gunakan __get_str;
        jika key adalah string, gunakan __get_id
        """
        if type(key) is int:
            return self.__get_str(key)
        elif type(key) is str:
            return self.__get_id(key)
        else:
            raise TypeError


class Weighting:
    @staticmethod
    def get_idf_weight(total_docs, term_df):
        return math.log10(total_docs / term_df)

    @staticmethod
    def get_bm25_tf_weight(k1, b, doc_tf, dl, avdl):
        return ((k1 + 1) * doc_tf) / (k1 * (1 - b + b * dl / avdl) + doc_tf)


def sorted_merge_posts_and_tfs(posts_tfs1, posts_tfs2):
    """
    Menggabung (merge) dua lists of tuples (doc id, tf) dan mengembalikan
    hasil penggabungan keduanya (TF perlu diakumulasikan untuk semua tuple
    dengn doc id yang sama), dengan aturan berikut:

    contoh: posts_tfs1 = [(1, 34), (3, 2), (4, 23)]
            posts_tfs2 = [(1, 11), (2, 4), (4, 3 ), (6, 13)]

            return   [(1, 34+11), (2, 4), (3, 2), (4, 23+3), (6, 13)]
                   = [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)]

    Parameters
    ----------
    list1: List[(Comparable, int)]
    list2: List[(Comparable, int]
        Dua buah sorted list of tuples yang akan di-merge.

    Returns
    -------
    List[(Comparablem, int)]
        Penggabungan yang sudah terurut
    """
    # TODO
    res = []
    i = j = 0

    while i < len(posts_tfs1) and j < len(posts_tfs2):
        if posts_tfs1[i][0] == posts_tfs2[j][0]:
            res.append((posts_tfs1[i][0], posts_tfs1[i][1] + posts_tfs2[j][1]))
            i += 1
            j += 1
        elif posts_tfs1[i] < posts_tfs2[j]:
            res.append((posts_tfs1[i][0], posts_tfs1[i][1]))
            i += 1
        else:
            res.append((posts_tfs2[j][0], posts_tfs2[j][1]))
            j += 1

    res = res + posts_tfs1[i:] + posts_tfs2[j:]
    return res


def preprocess_text(text):
    """
    Digunakan sebelum memproses query saat retrieval dan document saat
    indexing.
    """
    # TODO
    text = text.lower()
    text = re.sub("\s+", " ", text)  # Remove excess whitespace
    text = re.sub("[^\w\s]", " ", text)  # Remove punctuations
    text = re.sub(r"\d+", "", text)  # Remove numbers

    text = word_tokenize(text)

    stops = set(stopwords.words("english"))
    text = [word for word in text if word not in stops]

    stemmer = SnowballStemmer("english")
    text = [stemmer.stem(word) for word in text]

    return text
