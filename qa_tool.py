import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Shared LLM instance with history support
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=GOOGLE_API_KEY,
)

# Store conversation as list of LangChain message objects
conversation_history = []

def qa_tool(question: str, rag=False, persist_directory="chroma_db"):
    global conversation_history

    if rag:
        # Step 1: Improve the user's question using the LLM
        prompt_improvement_instruction = (
            "You are a helpful assistant that corrects spelling and grammar. "
            "Please improve the following sentence by correcting any spelling mistakes and grammatical errors. "
            "Only return the corrected sentence, without any additional commentary:\n"
        )
        improved_question_response = llm.invoke(prompt_improvement_instruction + question)
        improved_question = improved_question_response.content.strip()

        print(f"\n✨ Original Question: '{question}'")
        print(f"✨ Improved Question: '{improved_question}'\n")

        # Step 2: Use the improved question for RAG
        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        retriever = vectordb.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # Pass the improved question to the RAG chain
        answer = qa_chain.run(improved_question)

        # Optionally store RAG interaction in history
        conversation_history.append(HumanMessage(content=question)) # Store original for context
        conversation_history.append(AIMessage(content=answer))
        return f"🔎 Answer (RAG): {answer}"

    else:
        # Add user message
        conversation_history.append(HumanMessage(content=question))

        # Get model reply with full history
        response = llm.invoke(conversation_history)

        # Store assistant response
        conversation_history.append(AIMessage(content=response.content))

        prefix = "💬 Answer (LLM only):"
        return f"{prefix} {response.content}"

if __name__ == "__main__":
    print("💬 Chat Started. Type your questions below. Use /bye to exit.\n")
    print("---")

    while True:
        question = input("❓ You: ").strip()
        if question.lower() in ["/bye", "bye", "exit", "quit"]:
            print("👋 Goodbye!")
            break

        # Always use RAG for demonstration of prompt improvement
        answer = qa_tool(question, rag=True)
        print(answer)
        print("\n---")
        print("💡 Tip: Ask another question or type /bye to exit.\n")
