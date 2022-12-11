import os
import re
import time

from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.http import HttpResponseNotFound, HttpResponseBadRequest
from django.shortcuts import render

from home.bsbi import BSBIIndex
from home.compression import VBEPostings
from home.util import preprocess_text


def index(request):
    return render(request, "index.html")


def search(request):
    if not "q" in request.GET:
        return HttpResponseBadRequest("Search query required")

    start_time = time.time()

    query = request.GET["q"]
    docs = get_serp(query)

    page_number = request.GET.get("page")
    page_obj = paginate(docs, page_number)

    context = {
        "query": query,
        "exe_time": round(time.time() - start_time, 2),
        "page_obj": page_obj,
    }
    return render(request, "results.html", context=context)


def paginate(objects, page_number, obj_per_page=10):
    paginator = Paginator(objects, obj_per_page)

    try:
        page_obj = paginator.page(page_number)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)
    return page_obj


def get_serp(query):
    BSBI_instance = BSBIIndex(
        data_dir=os.path.join("home", "collection"),
        postings_encoding=VBEPostings,
        output_dir=os.path.join("home", "indices"),
    )

    docs = []

    for (_, doc_path) in BSBI_instance.retrieve_bm25(query, k=100):
        doc_id = re.search(r".*\\.*\\.*\\(.*)\.txt", doc_path).group(1)

        try:
            f = open(doc_path)
        except FileNotFoundError:  # Linux
            doc_path = doc_path.replace("\\", "/")
            f = open(doc_path)

        raw_title = f.readline().strip()
        title = get_title(raw_title)

        raw_content = f.read()
        clean_query = preprocess_text(query)
        content = get_content(raw_content, clean_query)

        docs.append(
            {
                "path": doc_path,
                "id": doc_id,
                "title": title,
                "content": content,
            }
        )

        f.close()
    return docs


def get_title(raw_title):
    title = re.sub(r"\d+. ", "", raw_title)  # clean numbers at page entry
    title = (title[:65]) if len(title) > 69 else title
    if title[-1] != ".":
        title += " ..."
    else:
        title = title[:-1]
    return title


def get_content(raw_content, clean_query):
    content = []

    # find all sentences containing ALL query words
    sentences = re.findall(r"([^.]*\.)", raw_content)
    for sentence in sentences:
        if all(word in sentence for word in clean_query):
            content.append(sentence)

    # else, for each query word, find a sentence containing the word
    if not content:
        for q in clean_query:
            try:
                match = re.findall(r"([^.]*?" + q + "[^.]*\.)", raw_content)[0]
                content.append(match)
            except IndexError:
                continue

    # else, get the first 165 letters of the content
    if not content:
        text = (raw_content[:161] + " ...") if len(raw_content) > 165 else raw_content
        content.append(text)

    content = " ... ".join(content)
    for q in clean_query:
        content = content.replace(q, "<b>" + q + "</b>")
    content = "<p>" + content + "</p>"
    return content

def view_doc(request, pk):
    pk = int(pk)
    if pk < 1 or pk > 1033:
        return HttpResponseNotFound("Document not found")

    block = pk // 100 + 1
    path = os.path.join("home", "collection", str(block), f"{pk}.txt")
    with open(path, "r") as f:
        content = f.read()

    context = {
        "pk": pk,
        "path": path,
        "content": content,
    }
    return render(request, "doc.html", context=context)
