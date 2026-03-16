# from datasets import Dataset
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy

from src.rag_pipeline import ScienceTeacherRAG
# import time


# # ------------------------------
# # Test Queries
# # ------------------------------

# test_questions = [
#     "How should I teach the concept of chemical reactions to Class 10 students?"
#     # "Suggest an experiment to demonstrate chemical reactions.",
#     # "What real-life examples explain chemical reactions?"
# ]

# rag = ScienceTeacherRAG()

# answers = []
# contexts = []

# print("Running RAG evaluation...\n")

# for question in test_questions:

#     answer, docs = rag.answer_query(question)

#     context_text = [doc.page_content for doc in docs]

#     answers.append(answer)
#     contexts.append(context_text)

#     print("Query:", question)
#     print("Answer:", answer[:150])
#     print("------")
#     time.sleep(3) 

# # ------------------------------
# # Build Evaluation Dataset
# # ------------------------------

# data = {
#     "question": test_questions,
#     "answer": answers,
#     "contexts": contexts
# }

# dataset = Dataset.from_dict(data)


# # ------------------------------
# # Run RAG Evaluation
# # ------------------------------

# result = evaluate(
#     dataset,
#     metrics=[
#         faithfulness,
#         answer_relevancy
#     ]
# )

# print("\nEvaluation Results:")
# print(result)

from src.rag_pipeline import ScienceTeacherRAG
from langchain_ollama import ChatOllama

rag = ScienceTeacherRAG()

question = "How should I explain Fleming’s right-hand rule to students?"

answer, docs = rag.answer_query(question)

context = "\n".join([d.page_content for d in docs])

# evaluation model
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

prompt = f"""
You are evaluating whether an answer is faithful to the provided context.

Context:
{context}

Question:
{question}

Answer:
{answer}

Score faithfulness, Answer relevancy from 0 to 1.

0 = answer contradicts context
1 = answer fully supported by context

Return ONLY the numbers.
"""

result = llm.invoke(prompt)

print("\nFaithfulness Score:", result.content)