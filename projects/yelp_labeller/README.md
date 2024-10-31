# Yelp labeller

## Структура

- [Лабораторная 1](./source/classifier)
- [Лабораторная 2](./source/association_meter)
- [Лабораторная 3](./source/vectorizer)
- [Лабораторная 4](./source/classify)
- [Лабораторная 5](./source/lab_5.ipynb)
- [Лабораторная 6](./source/lab_6.ipynb)

## Описание

Данный проект предназначен для обработки датасета и дальнейшего определения рейтинга отзыва по его тексту.

Проект состоит из модуля `classifier`, в котором:

1. файл `__main__.py` - содержит реализацию интерфейса командной строки;
2. файл `main.py` - содержит код для обработки датасета и сохранения его в tsv формате;
3. файл `tokenizer.py` - содержит код для токенизации текста, включая обработку сложных случаев (e-mail, номера
   телефонов, сокращения).

## Настройка окружения

Для создания окружения и установки зависимостей требуется выполнить следующую команду из корневой директории проекта :

```sh
pip install -r ./requirements.txt
```

## Запуск проекта

```sh
python main.py
```

## Запуск тестов

Для системы разработан набор модульных тестов, позволяющих оценить корректность генерируемых результатов. Для запуска
тестов используется следующая команда:

```sh
PYTHONPATH=source python -m unittest
```

## Запуск лаба 1

```sh
python main.py
```

## Запуск лаба 2

```sh
python -m association_meter.main
```

или

```sh
python .\projects\yelp_labeller\source\association_meter\main.py
```

## Запуск лаба 3

```sh
python -m vectorizer.main
python -m vectorizer.save_emb_lib
```

## Запуск лаба 4

```sh
python -m classify.main
```

## Запуск лаба 5

```sh
python -m notebook
```
