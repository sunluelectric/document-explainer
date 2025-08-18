import os
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import fitz
from bs4 import BeautifulSoup
import tiktoken
from agents import Agent, RunContextWrapper, function_tool, Runner, output_guardrail, trace, WebSearchTool, ModelSettings, GuardrailFunctionOutput
import asyncio

explainer = None

LLM_MODEL = "gpt-5-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
DOC_PATH = "./docs"
TOP_N_DEFAULT = 10
TOP_N_MAX = 25
SEMANTIC_SEARCH_MAX = 5
EMBEDDING_MAX_TOKENS = 2000
EMBEDDING_OVERLAP = 200

class DocumentExplainerOnlineCheck:
    def __init__(self):
        self.top_n = TOP_N_DEFAULT
        self.semantic_search_num = 0
        load_dotenv(override=True)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.create_client()
        self.set_tokenizer()
        self.retrieve_chunks()
        self.define_cross_check_agent()
        self.define_search_agent()
        
        
    def create_client(self):
        self.client = OpenAI(api_key=self.openai_api_key)

    def set_tokenizer(self):
        self.tokenizer = tiktoken.encoding_for_model(EMBEDDING_MODEL)

    def tokenize(self, text):
        return self.tokenizer.encode(text)
    
    def count_tokens(self, text):
        return len(self.tokenize(text))
    
    def chunk_text(self, text):
        words = text.split()
        chunks = []
        chunk = []
        tokens_so_far = 0
        for word in words:
            token_count = self.count_tokens(word)
            if tokens_so_far + token_count > EMBEDDING_MAX_TOKENS:
                chunks.append(" ".join(chunk))
                if EMBEDDING_OVERLAP > 0:
                    chunk = chunk[-EMBEDDING_OVERLAP:]
                    tokens_so_far = self.count_tokens(" ".join(chunk))
                else:
                    chunk = []
                    tokens_so_far = 0
            chunk.append(word)
            tokens_so_far += token_count
        if chunk:
            chunks.append(" ".join(chunk))

        return chunks
    
    def chunk_documents(self):
        chunks = []
        for filename in os.listdir(DOC_PATH):
            if filename.endswith('.pdf'):
                doc_path = os.path.join(DOC_PATH, filename)
                doc = fitz.open(doc_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                chunks.extend(self.chunk_text(text))
            elif filename.endswith('.html') or filename.endswith('.htm'):
                with open(os.path.join(DOC_PATH, filename), 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    text = soup.get_text()
                    chunks.extend(self.chunk_text(text))
        return chunks
    
    def generate_embeddings(self, chunks):
        embeddings = []
        for chunk in chunks:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=chunk,
                encoding_format="float"
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        return embeddings

    def save_chunks_embeddings(self, chunks, embeddings):
        with open('parsed_chunks.json', 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        np.save('embeddings.npy', embeddings)

    def load_chunks_embeddings(self):
        if os.path.exists('parsed_chunks.json') and os.path.exists('embeddings.npy'):
            with open('parsed_chunks.json', 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            embeddings = np.load('embeddings.npy')
            return chunks, embeddings
        return None, None
    
    def retrieve_chunks(self):
        self.chunks, self.embeddings = self.load_chunks_embeddings()
        if self.chunks is None or self.embeddings is None:
            self.chunks = self.chunk_documents()
            self.embeddings = self.generate_embeddings(self.chunks)
            self.save_chunks_embeddings(self.chunks, self.embeddings)
        print(f"SYSTEM: Retrieved {len(self.chunks)} chunks and {len(self.embeddings)} embeddings from the documents.")

    @staticmethod
    @function_tool
    def semantic_search(query: str):
        """Perform semantic search on the document chunks with the specified query."""
        print(f"SYSTEM: Performing semantic search with query: {query}")
        instance = explainer
        if instance.semantic_search_num >= SEMANTIC_SEARCH_MAX:
            return {"status": "error", "message": f"Maximum number of semantic searches ({SEMANTIC_SEARCH_MAX}) reached."}
        similarities = []
        query_embedding = instance.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
            encoding_format="float"
        ).data[0].embedding
        for embedding in instance.embeddings:
            similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
            similarities.append(similarity)
        top_indices = np.argsort(similarities)[::-1][:instance.top_n]
        top_chunks = [instance.chunks[i] for i in top_indices]
        context = "\n\n".join(top_chunks)
        instance.semantic_search_num += 1
        if not context:
            return {"status": "error", "message": "No relevant chunks found for the semantic search key."}
        else:
            return {"status": "success", "message": f"Semantic search completed successfully and {instance.top_n} chunks retrieved.", "data": context}
        

    @staticmethod
    @function_tool
    def request_increasing_top_n():
        """Request to increase the number of chunks returned."""
        print(f"SYSTEM: Requesting to increase the number of chunks.")
        instance = explainer
        instance.top_n += 5
        if instance.top_n > TOP_N_MAX:
            instance.top_n = TOP_N_MAX
            return {"status": "error", "message": f"Maximum value of top_n {TOP_N_MAX} reached and cannot be increased further."}
        return {"status": "success", "message": f"Maximum number of chunks returned increased to {instance.top_n}."}

    @staticmethod
    @function_tool
    def record_unknown_question(question: str):
        """Record an unknown question, if the relevant information is missing from the chunks."""
        with open('unknown_questions.json', 'a', encoding='utf-8') as f:
            json.dump({"question": question}, f, ensure_ascii=False, indent=2)
            f.write('\n')
        return {"status": "success", "message": "Question recorded."}

    @staticmethod
    @function_tool
    def record_suggestion(suggestion: str):
        """Record a suggestion for improving the document or the search process."""
        with open('suggestions.json', 'a', encoding='utf-8') as f:
            json.dump({"suggestion": suggestion}, f, ensure_ascii=False, indent=2)
            f.write('\n')
        return {"status": "success", "message": "Suggestion recorded."}

    def define_search_agent(self):
        system_prompt = (
            f"You are a helpful document explainer assistant.\n"
            f"Your task is to answer the user's questions based on the information recorded in the documents.\n"
            f"You can use tools to access the documents. You are allowed to perform semantic searches on the documents, and you may perform multiple searches with different query contents. You are also allowed to increase the number of returned chunks when necessary.\n"
            f"Always provide concise and accurate responses to the user's questions based on the documents.\n\n"
            f"Use the following tools when appropriate:\n"
            f"- record_unknown_question: Use this tool to record any question that cannot be answered from the chunks, even after you have performed several searches.\n"
            f"- record_suggestion: Use this tool to record suggestions for enriching the system, such as adding more documents or improving the search functionality.\n"
            f"- semantic_search: Use this tool to query the documents for further information. You can generate your own queries when needed. You may perform at most {SEMANTIC_SEARCH_MAX} searches per user request.\n"
            f"- request_increasing_top_n: Use this tool to request a larger number of chunks returned from semantic search. You may request this multiple times, with each increase adding 5 chunks, capped at {TOP_N_MAX}.\n\n"
            f"Guidelines:\n"
            f"- Always provide a clear and concise answer.\n"
            f"- Use tools only when necessary.\n"
            f"- If the user's question is unclear or ambiguous, ask for clarification.\n"
            f"- Always try to answer based on the information in the documents. For this reason, you are encouraged to perform at least one semantic search per user query.\n"
            f"- If you cannot find the answer in the returned chunks, you are encouraged to use semantic_search or request_increasing_top_n before concluding, until the limits are reached or you believe no further relevant information can be found.\n"
            f"- Do not add your own knowledge unless explicitly asked to, or if you believe the documents are lacking and your knowledge can meaningfully improve the answer. When you do so, make it clear to the user.\n"
            f"- You may chat with the user without accessing the documents only if the user's message is not a question (e.g., a suggestion for the system, or a request to summarize the conversation history).\n\n"
            f"You are also given a handoff that checks your response.\n"
            f"Always use the handoff to check the response. \n\n"
        )
        self.search_agent = Agent(
            name="Search agent",
            instructions=system_prompt,
            model=LLM_MODEL,
            tools=[
                self.record_unknown_question,
                self.record_suggestion,
                self.semantic_search,
                self.request_increasing_top_n
            ],
            handoffs=[self.cross_check_agent]
        )

    def define_cross_check_agent(self):
        system_prompt = (
            f"You are a helpful information cross-checking assistant.\n"
            f"Your upstream is an information retrieval and document explainer system that answers user queries using user documents and its built-in knowledge.\n"
            f"Your task is to cross-check whether the assistant's response contains any facts that are obviously incorrect or misleading.\n"
            f"You are allowed to use a tool to perform a web search. You can use this tool only once. Decide what is most likely to be wrong and formulate your own search query accordingly.\n\n"
            f"Do not alter the content from your upstream. Instead, append your comments at the end of the response, starting with [Cross-Check Agent Comment], so the user knows clearly that this part comes from your online verification, not from the documents.\n"
            f"If everything looks correct, or indeed is correct, just say so in your comments.\n"
            f"Since you can only perform one online check, if you suspect something is wrong but cannot find evidence or did not have the chance to verify it online, point that out as well.\n\n"
            f"Guidelines:\n"
            f"- Do not modify the content from your upstream. Copy-and-paste whatever your upstream gives you in your output. Only append your comments, and always prefix them with [Cross-Check Agent Comment].\n"
            f"- No need to explain the context. Directly starts with bullet points to state what is wrong and what should be the corrected answer. Simply use INCORRECT: and CORRECTED: to start the bullet points."
            f"- If everything looks correct according to your own knowledge, do not perform the web search. Just state that it looks correct.\n"
            f"- If something is incorrect or suspicious, perform the web search. You must come up with your own query to check the relevant information.\n"
            f"- You may perform only one web search.\n"
            f"- In [Cross-Check Agent Comment], specify what was wrong and what the correct information is according to the web search.\n"
            f"- If you suspect something is incorrect but cannot verify it from the search results, include your suspicions in [Cross-Check Agent Comment] and make it clear that they are based on your own knowledge, not on online evidence."
        )
        self.cross_check_agent = Agent(
            name="Cross-check agent",
            instructions=system_prompt,
            model=LLM_MODEL,
            tools=[WebSearchTool(search_context_size="low")],
            model_settings=ModelSettings(
                parallel_tool_calls=False
            )
        )

    async def chat(self, query, history):
        messages = history + [{"role": "user", "content": query}]
        response = await Runner.run(self.search_agent, messages)
        return response.final_output
    
    def save_history(self, history):
        with open('history.json', 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def load_history(self):
        if os.path.exists('history.json'):
            with open('history.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def main(self):
        history = self.load_history()
        while True:
            query = input("You: ")
            if query.lower() in ['exit', 'quit']:
                break
            response = asyncio.run(self.chat(query, history))
            self.top_n = TOP_N_DEFAULT
            self.semantic_search_num = 0
            print(f"Bot: {response}")
            print("---")
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})
            self.save_history(history)

if __name__ == "__main__":
    explainer = DocumentExplainerOnlineCheck()
    explainer.main()