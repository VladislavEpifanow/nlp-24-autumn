# Лабораторная работа №5 (Поиск по векторной БД)


## Задание:

Необходимо записать ваш датасет в векторную базу данных и выполнить эксперименты по поиску схожих фрагментов текста, соответствующих запросу. 

1. Разбиение текстовых документов на фрагменты.
   1. Разработать алгоритм разбиения текстовых документов на фрагменты текста. Можно использовать уже существующие механизмы, например, разбиение по длине фрагмента текста в символах и пересечению с соседними фрагментами. Дополнительные баллы за усложненные варианты, например: учитывать границы токенов и предложений.
   2. Подготовка метаданных для каждого фрагмента, таких как класс документа, автор документа, ключевые слова и др.
2. Векторизация фрагментов текста. 
   В качестве метода векторизации можно использовать стороний api (huggingface, openai, etc.), w2v или любую другую модель на выбор. Рекомендуется применить модель с huggingface. Подходящие модели с [huggingface](https://huggingface.co/models?pipeline_tag=sentence-similarity) можно выбрать по ссылке. Примеры моделей из huggingface: 
   1. ai-forever/sbert_large_mt_nlu_ru (Russian language)
   2. sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (50 languages)
3. Создание Векторной Базы Данных (ВБД).
   1. Необходимо выбрать одну из доступных ВБД, например: Chroma (рекомендуемая с точки зрения простоты), Pinecone и т.д.
   2. Реализовать механизм загрузки и сохранения текстовых данных в ВБД.
4. Поиск схожих фрагментов текста
   1. Выбрать алгоритмы similarity для поиска схожих фрагментов текста.
   2. Реализовать механизм поиска схожих фрагментов по заданным запросам.
5. Оценка качества поиска 
   1. Сгенерировать набор запросов к ВБД. 
   2. Провести оценку качества поиска, определяя, насколько хорошо схожие фрагменты отображаются в результатах поиска. Оценку можно выполнить следующими способами:
      1. на основе ручной оценки качества запросов и соответствующих ответов;
      2. посчитать средний порядковый номер требуемого фрагмента в отсортированном по релевантности спике результатов. 

Дополнительные баллы: провести эксперименты с разными системами векторизации и алгоритмами similarity. Сравнить средний порядковый номер требуемого фрагмента в отсортированном по релевантности спике результатов.

Общие рекомендации при использовании Google Colab:
- подключаться к окружению с GPU (T4);
- сохранять индекс на google disk, чтобы каждый раз его не пересоздавать.

Примеры классов, которые могут потребоваться для выполнения данного задания описаны в [ноутбуке](https://colab.research.google.com/drive/1XywdDFIza0iu4HX47e7HaXkEO3T-VBRr#scrollTo=tcb0TE2y9o0S):

```python
class Loader:
    def load_single_document(self, file_path: str):
      pass

    def load_documents(self, source_dir: str):
      pass

    
class Splitter:
    def __init__(self, chunk_size, chunk_overlap):
        pass

    def split_documents(self, documents):
        pass

    
class Collector:
    def add(self, texts: list[str], metadatas: list[dict]):
      pass

    def add_from_directory(self, dir_path: str):
      pass

    def get(self, search_strings: list[str], n_results: int) -> list[Document]:
      pass

    def get_documents(self, search_string: str, n_results: int, score_threshold: float) -> list[Document]:
      pass

    def clear(self):
      pass

    
class Embedder():
  def __init__(self):
    pass

  def get_embedder(self):
    pass

  
class HuggingFaceEmbedder(Embedder):
    pass


class ChromaCollector(Collector):
    pass


class CollectorEvaluator:
  def __init__(self, collector: Collector, n_top=100):
    pass

  def explore_collector(self, text):
    pass

  def eval(self, query, answer):
    pass

  def calculate_statistics(self, data):
    pass

  def explore_and_calculate(self, data):
    pass
```

