
from sentence_transformers import SentenceTransformer

class Embedder():
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    def __call__(self, input):
        input = self.model.encode(input).tolist()
        return input

import chromadb
import numpy as np
from tqdm import tqdm
class CustomChromaDB():
    def __init__(self):
        # Инициализация клиента ChromaDB с указанием пути и модели для эмбеддингов
        self.embedding_model = Embedder()
        self.db_client = chromadb.PersistentClient(path='/content/drive/MyDrive/мага 1 сем/Database')
        self.data_collection = self.db_client.get_or_create_collection(name="custom_coll", embedding_function=self.embedding_model)

    def upload_data(self, fragments, metadata, batch_size=20000):
        # Генерация уникальных идентификаторов для фрагментов
        fragment_ids = [str(index) for index in range(len(metadata))]
        total_batches = len(fragments) // batch_size
        end = 0
        for batch_idx in tqdm(range(total_batches)):
            start = batch_idx * batch_size
            end = start + batch_size
            batch_fragments = fragments[start:end]
            batch_metadata = metadata[start:end]
            batch_ids = fragment_ids[start:end]

            # Добавление фрагментов в коллекцию
            self.data_collection.add(
                documents=batch_fragments,
                embeddings=self.embedding_model(batch_fragments),
                metadatas=batch_metadata,
                ids=batch_ids
            )

        # Добавление оставшихся данных, если их размер меньше batch_size
        remaining_fragments = fragments[end:]
        if remaining_fragments:
            self.data_collection.add(
                documents=remaining_fragments,
                embeddings=self.embedding_model(remaining_fragments),
                metadatas=metadata[end:],
                ids=fragment_ids[end:]
            )

        print("Dataset successfully uploaded to ChromaDB.")


    def search(self, text, count = 1):
        vector = self.embedding_model(text)
        result = self.data_collection.query(
            query_embeddings = vector,
            n_results = count,
            include=['distances','embeddings', 'documents', 'metadatas'],
        )
        return result

import requests
import json

def make_post_request(text):
    url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key='

    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": text
                    }
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response)
    if response.status_code == 200:
        print(response.json())
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")




import gradio as gr
from evaluate import load
bertscore = load("bertscore")

def echo(message, history):
    result = cdb.search(message, 5)
    prompt = f"Context: {' '.join(result['documents'][0])}." + f"Question: {message}"
    answer = make_post_request(prompt)

    return f"{answer}"

#print(echo('What is your favorite color?', ''))

def evaluate_question(question, desired_answer):
    result = cdb.search(question, 3)
    prompt = f"Context: {' '.join(result['documents'][0])}." + f"Question: {question}"
    answer = make_post_request(prompt)
    print(answer)
    metric = bertscore.compute(predictions=[answer], references=[desired_answer], model_type="distilbert-base-uncased")
    print(metric)
    return f"{answer}"

evaluate_question('What mean Ezekiel when said demolish your towers?', 'that clearly implied that the walls would still be standing so people would know where the towersused to be. ')


ex = ['Do you love ITMO', 'How work Mac sound hardware', 'If you get teh IIvx ->C650 upgrade, does it include a new sticker to cover the IIvx identifier with a Centris 650 indetifier?' ]
demo = gr.ChatInterface(fn=echo, examples=ex, title="Echo Bot")
demo.launch(share=True)