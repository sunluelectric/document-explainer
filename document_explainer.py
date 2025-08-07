import os
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import fitz
from bs4 import BeautifulSoup
import tiktoken

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

record_suggestion_json = {
    "name": "record_suggestion",
    "description": "Use this tool to record a suggestion for enriching the system, such as adding more documents or improving the search functionality",
    "parameters": {
        "type": "object",
        "properties": {
            "suggestion": {
                "type": "string",
                "description": "The suggestion for enriching the system"
            },
        },
        "required": ["suggestion"],
        "additionalProperties": False
    }
}

request_more_info_json = {
    "name": "request_more_info",
    "description": "Use this tool to request larger number of results from the semantic search",
    "parameters": {
        "type": "object",
        "properties": {
            "is_more_results_required": {
                "type": "boolean",
                "description": "Whether to return more results"
            }
        },
        "required": ["is_more_results_required"],
        "additionalProperties": False
    }
}

class DocumentExplainer:
    def __init__(self):
        self.load_env()
        self.create_client()
        self.set_tokenizer()
    
    def load_env(self):
        load_dotenv(override=True)
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.LLM_MODEL = "gpt-4.1-mini"
        self.LLM_EMBEDDING_MODEL = "text-embedding-3-small"
        self.DOC_DIR_PATH = "./docs"
        self.TOP_N_DEFAULT = 10
        self.TOP_N = self.TOP_N_DEFAULT
        self.TOP_N_MAX = 50
        self.MAX_TOKENS = 2000
        self.OVERLAP = 200

    def create_client(self):
        self.client = OpenAI(api_key=self.OPENAI_API_KEY)

    def set_tokenizer(self):
        self.tokenizer = tiktoken.encoding_for_model(self.LLM_EMBEDDING_MODEL)

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
            if tokens_so_far + token_count > self.MAX_TOKENS:
                chunks.append(" ".join(chunk))
                if self.OVERLAP > 0:
                    chunk = chunk[-self.OVERLAP:]
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
        for filename in os.listdir(self.DOC_DIR_PATH):
            if filename.endswith('.pdf'):
                doc_path = os.path.join(self.DOC_DIR_PATH, filename)
                doc = fitz.open(doc_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                chunks.extend(self.chunk_text(text))
            elif filename.endswith('.html') or filename.endswith('.htm'):
                with open(os.path.join(self.DOC_DIR_PATH, filename), 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    text = soup.get_text()
                    chunks.extend(self.chunk_text(text))
        return chunks
    
    def generate_embeddings(self, chunks):
        embeddings = []
        for chunk in chunks:
            response = self.client.embeddings.create(
                model=self.LLM_EMBEDDING_MODEL,
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

    def semantic_search(self, query, embeddings, chunks, top_n=0):
        similarities = []
        query_embedding = self.client.embeddings.create(
            model=self.LLM_EMBEDDING_MODEL,
            input=query,
            encoding_format="float"
        ).data[0].embedding
        for embedding in embeddings:
            similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
            similarities.append(similarity)
        top_indices = np.argsort(similarities)[::-1][:top_n]
        top_chunks = [chunks[i] for i in top_indices]
        return top_chunks

    def save_history(self, history):
        with open('history.json', 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def load_history(self):
        if os.path.exists('history.json'):
            with open('history.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def record_unknown_question(self, question):
        with open('unknown_questions.json', 'a', encoding='utf-8') as f:
            json.dump({"question": question}, f, ensure_ascii=False, indent=2)
            f.write('\n')
        return {"status": "success", "message": "Question recorded."}
    
    def record_suggestion(self, suggestion):
        with open('suggestions.json', 'a', encoding='utf-8') as f:
            json.dump({"suggestion": suggestion}, f, ensure_ascii=False, indent=2)
            f.write('\n')
        return {"status": "success", "message": "Suggestion recorded."}
    
    def request_more_info(self, is_more_results_required):
        if is_more_results_required:
            self.TOP_N += 5
            if self.TOP_N > self.TOP_N_MAX:
                self.TOP_N = self.TOP_N_MAX
                return {"status": "error", "message": f"Cannot increase TOP_N beyond {self.TOP_N_MAX}."}
        return {"status": "success", "message": f"Top N increased to {self.TOP_N}."}
    
    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            
            # Use getattr to get the method from self instead of globals()
            tool = getattr(self, tool_name, None)
            result = tool(**arguments) if tool else {"status": "error", "message": f"Tool {tool_name} not found"}
            results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
        return results
    
    def tools(self):
        return [
            {"type": "function", "function": record_unknown_question_json},
            {"type": "function", "function": record_suggestion_json},
            {"type": "function", "function": request_more_info_json}
        ]
    
    def system_prompt(self):
        system_prompt = (
            f"You are a document explainer system. You are asked to explain variety of documents that is saved in the user's system.\n"
            f"Semantic search has been implemented prior to this request. The most relevant chunks have been identified. These chunks will be given to you shortly. The total number of chunks will also be given. \n"
            f"Your task is to provide a concise and accurate explanation of the document based on the provided chunks and the user's questions. \n"
            f"Use the following tools when necessary:\n"
            f"- record_unknown_question: Use this tool to record any question that couldn't be answered from the chunks, even after you have requested the maximum number of chunks.\n"
            f"- record_suggestion: Use this tool to record a suggestion for enriching the system, such as adding more documents or improving the search functionality.\n"
            f"- request_more_info: Use this tool to request larger number of chunks returned from the semantic search. Notice that there is a limit on the number of {self.TOP_N_MAX} chunks that can be returned. Do not use this function to request more chunks if that limit is hit.\n"
            f"Remember to always provide a clear and concise explanation, and use the tools only when necessary.\n"
            f"If you cannot find the answer in the chunks, let the user know honestly, especially if the user requires you to answer based on the chunks.\n"
            f"If you cannot find the answer in the chunks, and you think you can answer based on your own knowledge, let the user know that you are answering based on your own knowledge.\n\n"
        )
        return system_prompt
    
    def chat(self, query, history):
        chunks, embeddings = self.load_chunks_embeddings()
        if chunks is None or embeddings is None:
            chunks = self.chunk_documents()
            embeddings = self.generate_embeddings(chunks)
            self.save_chunks_embeddings(chunks, embeddings)
        context = self.semantic_search(query, embeddings, chunks, top_n=self.TOP_N)
        intro_prompt = self.system_prompt() + (
            f"Below are information chunks extracted from various documents.\n"
            f"Current number of chunks: {len(chunks)}.\n\n"
        )
        content_prompt = intro_prompt + "\n\n".join(context)
        messages = (
            [{"role": "system", "content": content_prompt}] + 
            history +
            [{"role": "user", "content": query}]
        )
        done = False
        tools = self.tools()
        while not done:
            response = self.client.chat.completions.create(model=self.LLM_MODEL, messages=messages, tools=tools)
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content
    
    def main(self):
        history = self.load_history()
        while True:
            query = input("You: ")
            if query.lower() in ['exit', 'quit']:
                break
            response = self.chat(query, history)
            self.TOP_N = self.TOP_N_DEFAULT
            print(f"Bot: {response}")
            print("---")
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})
        self.save_history(history)

if __name__ == "__main__":
    explainer = DocumentExplainer()
    explainer.main()