import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from src.rag_pipeline import ScienceTeacherRAG
from dotenv import load_dotenv

load_dotenv()

def main():

    rag = ScienceTeacherRAG()

    print("\nScience Teaching Assistant Ready!\n")

    while True:

        query = input("Ask a teaching question (or type exit): ")

        if query.lower() == "exit":
            break

        answer, docs = rag.answer_query(query)

        print("\nAnswer:\n")
        print(answer)
        print("\nRetrieved Context:\n")
        for i, doc in enumerate(docs):
            print(f"Context {i+1} (Page {doc.metadata['page']}):\n")
            print(doc.page_content[:300])
            print("\n-----------------\n")


if __name__ == "__main__":
    main()