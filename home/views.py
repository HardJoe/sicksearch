import os
import re

from django.shortcuts import render

from home.bsbi import BSBIIndex
from home.compression import VBEPostings


def index(request):
    return render(request, 'index.html')


def search(request):
    if 'q' in request.GET:
        query = request.GET['q']
        docs = get_serp(query)
        context = {'query': query, 'docs': docs}
    return render(request, 'results.html', context=context)


def get_serp(query):
    BSBI_instance = BSBIIndex(
        data_dir=os.path.join("home", "collection"), postings_encoding=VBEPostings, output_dir=os.path.join("home", "indices")
    )

    docs = [{'path': doc} for (_, doc) in BSBI_instance.retrieve_bm25(query, k=10)]
    for doc in docs:
        with open(doc['path']) as f:
            title = f.readline()
            title = re.sub(r'\d+. ', '', title)
            title = (title[:65] + ' ...') if len(title) > 69 else title
            doc['title'] = title

            content = f.read()
            content = (content[:161] + ' ...') if len(content) > 165 else content
            doc['content'] = content
    return docs
