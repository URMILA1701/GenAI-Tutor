from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


from src.config import VECTOR_DB_PATH, EMBEDDING_MODEL, TOP_K_RESULTS



class ScienceTeacherRAG:

    def __init__(self):

        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        self.vector_db = FAISS.load_local(
            VECTOR_DB_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        self.top_k = TOP_K_RESULTS


        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0.3
        )


    def answer_query(self, query):

        results = self.vector_db.similarity_search_with_score(
            query,
            k=self.top_k
        )
        filtered_docs = []
        for doc, score in results:
            if score < 0.6:   
                filtered_docs.append(doc)

        if len(filtered_docs) == 0:
            filtered_docs = [doc for doc, _ in results]
        
        docs = filtered_docs

        textbook_context = "\n".join([d.page_content for d in docs])


        context = f"""
Textbook Reference:
{textbook_context}

"""

        prompt = f"""
You are an AI assistant helping a secondary school science teacher prepare lessons using the NCERT Class 10 Science textbook.

IMPORTANT RULES:
- Use ONLY the information provided in the "Knowledge Base Context".
- Do NOT add information that is not present in the context.
- If the context does not contain enough information, say so.

Step 1: Understand the teacher's question and determine the most suitable teaching task out of below tasks.
Possible tasks:
- Lesson Introduction
- Concept Explanation
- Interesting Facts
- Classroom Activity / Experiment
- Real-world Examples
- Lesson Plan

Step 2: Use the retrieved textbook context to generate the response.
Knowledge Base Context:
{context}

Teacher Question:
{query}

Output Only the following format:
Task Identified: <task>
Answer:
<teacher-friendly explanation>
Citations:
<quote the exact sentences or phrases from the knowledge base that support your answer>
"""

        response = self.llm.invoke(prompt)

        return response.content,docs