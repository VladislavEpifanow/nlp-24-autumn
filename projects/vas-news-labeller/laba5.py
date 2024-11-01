import re
import os
from tqdm import tqdm

def divide_text_with_overlap(text, part_size=100, overlap_size=20):
    parts = []
    position = 0
    while position < len(text):
        section = text[position:position + part_size]
        if len(section) == part_size:
            parts.append(section)
        position += part_size - overlap_size  # Шаг с учетом пересечения
    return parts

def retrieve_author_info(content):
    author_match = re.search(r'From:(.*?)(?=\w+:|$)', content, re.DOTALL)
    if author_match:
        return author_match.group(1).strip()
    return "Unknown"

# Основная функция обработки набора данных
def handle_dataset(path_to_dataset, part_size=100):
    text_fragments = []
    meta_data = []
    folder_list = os.listdir(path_to_dataset)

    for folder_name in tqdm(folder_list):
        current_folder = os.path.join(path_to_dataset, folder_name)
        file_list = os.listdir(current_folder)

        for file_name in file_list:
            file_location = os.path.join(current_folder, file_name)
            with open(file_location, 'r', encoding='latin1') as opened_file:
                raw_text = opened_file.read()
                processed_text = re.sub(r'[^\w\s.,!?-]', '', raw_text.replace('\t', ' ').replace('\n', ' ')) #убираем знаки
                processed_text = re.sub(r'\s+', ' ', processed_text)  # Убираем лишние пробелы
                sections = divide_text_with_overlap(processed_text, part_size)
                found_author = retrieve_author_info(raw_text)

                for fragment_id, section in enumerate(sections):
                    metadata_entry = {
                        'category': folder_name,
                        'fragment_id': f'{file_name}-{fragment_id}',
                        'author': found_author
                    }
                    meta_data.append(metadata_entry)
                    text_fragments.append(section)

    return text_fragments, meta_data

# Путь к данным
dataset_directory = '/content/20news-bydate-train'
fragments, metadata = handle_dataset(dataset_directory)



from sentence_transformers import SentenceTransformer

class Embedder():
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    def __call__(self, input):
        input = self.model.encode(input).tolist()
        return input

# embedder = Embedder()


import chromadb
import numpy as np
from tqdm import tqdm
class CustomChromaDB():
    def __init__(self):
        # Инициализация клиента ChromaDB с указанием пути и модели для эмбеддингов
        self.embedding_model = Embedder()
        self.db_client = chromadb.PersistentClient(path='/content/drive/MyDrive/мага 1 сем/ChromaDB')
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
    

cdb = CustomChromaDB()
cdb.upload_data(fragments, metadata)

cdb = CustomChromaDB()
questions = [
    ['Do you have a copy?', 1],
    ['Have you seen a local news?', 1],
    ['What do you know about computer science?', 2],
    ['Can you give me some information about space?']
]
count = 0
for question in questions:
    result = cdb.search(question[0], question[1])
    print(count)
    print(question[0])
    print(result['documents'])
    print()
    count +=1