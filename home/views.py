import os
import re
import time

from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import render

from home.bsbi import BSBIIndex
from home.compression import VBEPostings


def index(request):
    return render(request, "index.html")


def search(request):
    if not "q" in request.GET:
        return HttpResponseBadRequest

    start_time = time.time()

    query = request.GET["q"]
    docs = get_serp(query)

    paginator = Paginator(docs, 10)  # 10 employees per page

    page_number = request.GET.get("page")

    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        # if page is not an integer, deliver the first page
        page_obj = paginator.page(1)
    except EmptyPage:
        # if the page is out of range, deliver the last page
        page_obj = paginator.page(paginator.num_pages)

    context = {
        "query": query,
        "exe_time": round(time.time() - start_time, 2),
        "page_obj": page_obj,
    }
    return render(request, "results.html", context=context)


def get_serp(query):
    BSBI_instance = BSBIIndex(
        data_dir=os.path.join("home", "collection"),
        postings_encoding=VBEPostings,
        output_dir=os.path.join("home", "indices"),
    )

    docs = []

    for (_, doc) in BSBI_instance.retrieve_bm25(query, k=100):
        docs.append(
            {
                "path": doc,
                "id": re.search(r".*\\.*\\.*\\(.*)\.txt", doc).group(1),
            }
        )

    for doc in docs:
        with open(doc["path"]) as f:
            title = f.readline()
            title = re.sub(r"\d+. ", "", title)
            title = (title[:65] + " ...") if len(title) > 69 else title
            doc["title"] = title

            content = f.read()
            content = (content[:161] + " ...") if len(content) > 165 else content
            doc["content"] = content

    return docs


def view_doc(request, pk):
    block = int(pk) // 100 + 1
    f = open(os.path.join("home", "collection", str(block), f"{pk}.txt"), "r")
    file_content = f.read()
    f.close()
    return HttpResponse(file_content, content_type="text/plain")
