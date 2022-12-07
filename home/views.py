from django.shortcuts import render

# import os

# from django.conf import settings

# from .bsbi import BSBIIndex
# from .compression import VBEPostings




def index(request):
    return render(request, 'index.html')

def search(request):
    # if 'q' in request.GET:
    #     query = request.GET['q']
    #     docs = get_serp(query)
    #     context = {'docs': docs}

    # return render(request, 'results.html', context=context)
    return render(request, 'results.html')



# def get_serp(query):
#     # sebelumnya sudah dilakukan indexing
#     # BSBIIndex hanya sebagai abstraksi untuk index tersebut
#     BSBI_instance = BSBIIndex(
#         data_dir="collection", postings_encoding=VBEPostings, output_dir=os.path.join('index', 'index')
#     )

#     docs = [{'path': doc} for (_, doc) in BSBI_instance.retrieve_bm25(query, k=10)]
#     for doc in docs:
#         with open(doc['path']) as f:
#             title = f.readline()
#             title = (title[:65] + ' ...') if len(title) > 69 else title
#             doc['title'] = title

#             content = f.read()
#             content = (content[:161] + ' ...') if len(content) > 165 else content
#             doc['content'] = content
#     return docs
