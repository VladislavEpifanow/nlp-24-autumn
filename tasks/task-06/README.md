# Лабораторная работа №6 (Question Answering)


## Задание:

Необходимо запустить и протестировать QA на основе LLM модели, можно выбрать любую LLM модель (рекомендуется искать на [huggingface](https://huggingface.co/models)):

1. на основе llama.cpp - можно запустить на GPU и/или на CPU, примеры:
   1. [TheBloke/Mistral 7B OpenOrca - GGUF](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF)
   3. [Saiga Mistral](https://huggingface.co/IlyaGusev/saiga_mistral_7b_gguf) (for Russian language)

2. на GPU, примеры:
   1. [Open-Orca/Mistral-7B-OpenOrca🐋](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca)
   2. [Saiga Mistral LORA](https://huggingface.co/IlyaGusev/saiga_mistral_7b_lora) (for Russian language)


При разработке QA необходимо поддержать следующий алгоритм:
1. Считать запрос/вопрос пользователя; 
2. Считать из БД top-N записей соответствующих запросу/вопросу пользователя;
3. Передать в LLM модель данные top-N записей в качестве контекста;
4. Сформировать ответ.

Необходимо написать не менее 10 пар запросов и соответствующих им ответов, далее оценить качество ответа. Оценку можно выполнить:
1. вручную по предложенному вами методу; 
2. одним из методов оценки QA систем, например [bertscore](https://huggingface.co/spaces/evaluate-metric/bertscore). Данный подход позволяет получить дополнительные баллы.

Пример реализации похожей задачи без оценки, но с [интерфейсом](https://huggingface.co/spaces/IlyaGusev/saiga_13b_llamacpp_retrieval_qa) можно посмотреть по данной ссылке
[Code](https://huggingface.co/spaces/IlyaGusev/saiga_13b_llamacpp_retrieval_qa/blob/main/app.py) 

Дополнительно можно добавить интерфейс на основе библиотеки [Gradio](https://www.gradio.app/guides/quickstart). [Простейший пример](https://www.gradio.app/docs/chatinterface) состоит из следующего кода:
```python
import gradio as gr

def echo(message, history):
    return message

demo = gr.ChatInterface(fn=echo, examples=["hello", "hola", "merhaba"], title="Echo Bot")
demo.launch()
```

И имеет следующую визуализацию:
![gradio visualization](gradio_example.png)

Общие рекомендации при использовании Google Colab:
- подключаться к окружению с GPU (T4);
- сохранять индекс на google disk, чтобы каждый раз его не пересоздавать;
- использовать модели 7b (mistral и т.п.).

Примеры классов, которые могут потребоваться для выполнения данного задания описаны в [ноутбуке](https://colab.research.google.com/drive/1wSNDNxGT0H0jhBRnatSWfD7chHXvGki7).
