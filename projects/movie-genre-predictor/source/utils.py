import numpy as np
import matplotlib.pyplot as plt

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