import os

from home.bsbi import BSBIIndex
from home.compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(
    data_dir=os.path.join("home", "collection"), postings_encoding=VBEPostings, output_dir=os.path.join("home", "indices")
)

queries = [
    "alkylated with radioactive iodoacetate",
    "psychodrama for disturbed children",
    "lipid metabolism in toxemia and normal pregnancy",
]
for query in queries:
    print("Query   : ", query)
    print("Results :")
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k=10):
        print(f"{doc:30} {score:>.3f}")
    print()

def test_search(query):
    print("Query   : ", query)
    print("Results :")
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k=10):
        print(f"{doc:30} {score:>.3f}")
    print()
