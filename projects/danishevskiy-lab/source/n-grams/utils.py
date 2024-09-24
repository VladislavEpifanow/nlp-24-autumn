import numpy as np
import matplotlib.pyplot as plt


def data_to_text(data, column='lemma'):
    return ' '.join(data[column].values)

def plot_dict_content(dictionary: dict):
    k,v = next(iter(dictionary.items()))
    if isinstance(k, tuple) and isinstance(k[0], str):
        arr = [(','.join(k),v) for k,v in dictionary.items()]
    else:
        arr = [(k,v) for k,v in dictionary.items()]
    arr = np.array(arr)
    plt.figure(figsize=(20,10))
    plt.plot(arr[:10, 0], np.array(arr[:10,1], dtype=int))
    plt.show()

def merge_files(file_list):
    def read_file(file_path):
        with open(file_path, 'r') as f:
            text = f.read()
        return text
    texts = []
    for filename in file_list:
        texts.append(read_file(filename))
    return '\n'.join(texts)
