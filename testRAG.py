from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline 
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import torch
import time
import faiss
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGHelper:
    def __init__(self, index_path, dataset_path):
        self.index_path = index_path
        self.dataset_path = dataset_path
        self.index = None
        self.dataset = None
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Loaded embedding model: all-MiniLM-L6-v2")
        
        self.load_resources()
    
    def load_resources(self):
        if os.path.exists(self.index_path) and os.path.exists(self.dataset_path):
            self.index = faiss.read_index(self.index_path)
            
            with open(self.dataset_path, 'rb') as f:
                self.dataset = pickle.load(f)
            print(f"RAG resources loaded: index contains {self.index.ntotal} vectors")
        else:
            print(f"Warning: RAG resources not found at {self.index_path} or {self.dataset_path}")
    
    def get_embedding(self, text):
        return self.embedding_model.encode(text)
    
    def retrieve(self, query, top_k=3):
        if self.index is None or self.dataset is None:
            print("RAG resources not loaded, returning empty context")
            return ""
        
        try:
            query_vector = self.get_embedding(query)
            
            # Reshape for FAISS query
            query_vector = np.array(query_vector).reshape(1, -1).astype('float32')
            
            # Search indexed version for relevant responses
            distances, indices = self.index.search(query_vector, top_k)
            
            # Retrieve resultant documents
            retrieved_docs = [self.dataset["documents"][idx] for idx in indices[0] if idx < len(self.dataset["documents"])]
            
            # Format the context
            context = "\n\n".join(retrieved_docs)
            print(f"Retrieved {len(retrieved_docs)} documents for context")
            return context
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return ""

class HFModelRunner:
    def __init__(self, model_id: str, max_new_tokens: int = 2048, temperature: float = 0.6):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        self.test_id = 1
        self.quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            quantization_config=self.quantization_config,
            attn_implementation="flash_attention_2",
        )

        self.model.gradient_checkpointing_enable()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            device_map="auto",
            torch_dtype="auto",
            truncation=False
        )

        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        
        self.output_parser = StrOutputParser()

    def create_prompt_template(self, template: str) -> PromptTemplate:
        return PromptTemplate.from_template(template)

    def generate_response(self, prompt_template: PromptTemplate, **kwargs) -> str:
        chain = prompt_template | self.llm | self.output_parser

        # check if same earlier input is being passed by checking earlier input from input.txt and finding that substring in the next input
        return chain.invoke(kwargs)

    def cleanup(self):
        torch.cuda.empty_cache()
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'pipe'):
            del self.pipe

def main():
    start_time = time.time()
    print("Starting model execution...")
    model_runner = HFModelRunner("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rag_helper = RAGHelper(
        os.path.join(current_dir, "rag_dataset.index"),
        os.path.join(current_dir, "rag_dataset.pkl")
    )
    
    template = """{prompt_text}
    
    Retrieved Context:
    {context}
    
    Game Text: {game_text}
    Evaluation: <think>\n"""

    prompt = model_runner.create_prompt_template(template)

    try:
        #Reading input from file to be evaluated
        with open("input.txt", "r") as file:
            input_game_text = file.read()

        
        #Final prompt from external file
        with open("prompt.txt", "r") as file:
            prompt_fin_text = file.read()
        
        #Configure input to LLM model
        main_context = rag_helper.retrieve(input_game_text + " " + prompt_fin_text)

        response = model_runner.generate_response(
            prompt,
            prompt_text=prompt_fin_text,
            game_text=input_game_text,
            context=main_context
        )
        
        parts = response.split("</think>", 1)
        
        if len(parts) > 1:
            with open("output.txt", "w") as output_file:
                output_file.write(parts[1].strip())
        else:
            print("Tag '</think>' was not found.")

    finally:
        #Calculating and printing elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution completed in {elapsed_time:.2f} seconds")
        
        #Append execution time to the output file
        with open("/home/ivan/.venv/output.txt", "a") as output_file:
            output_file.write(f"\n\nExecution completed in {elapsed_time:.2f} seconds")

        #Clearing the memory
        model_runner.cleanup()

if __name__ == "__main__":
    main()
