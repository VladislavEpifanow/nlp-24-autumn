import numpy as np
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from numpy.linalg import norm
from sklearn.decomposition import PCA


def cosine(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def get_token_vector(token, w2v):
    if token in w2v.wv:
        return w2v.wv[token]
    else:
        return np.zeros(w2v.vector_size)


def get_words_cosine(w2v_model: Word2Vec, words: list[str]):
    main_word = words[0]
    main_emb = get_token_vector(main_word, w2v_model)
    cosine_dict = {main_word: {}}
    vector_dict = {main_word: main_emb}
    for word in words[1:]:
        word_emb = get_token_vector(word, w2v_model)
        cosine_val = cosine(main_emb, word_emb)
        cosine_dict[main_word][word] = cosine_val
        vector_dict[word] = word_emb
    return cosine_dict, vector_dict


def get_cosine_distance_groups(w2v):
    words_g1 = ["like", "love", "hate"]
    words_g2 = ["good", "best", "nice", "bad", "worst", "terrible", "awful"]
    words_g3 = ["beer", "water", "drink", "mojito", "window", "dr", "weather"]

    cosine_dict = {}
    vector_list = []
    for group in [words_g1, words_g2, words_g3]:
        group_cosine, vector_dict = get_words_cosine(w2v, words=group)
        cosine_dict.update(group_cosine)
        vector_list.append(vector_dict)
    return cosine_dict, vector_list


def save_cosine_distance(cosine_dict):
    with open("cosine_distance.txt", "w") as file:
        for word, dists in cosine_dict.items():
            file.write(word + "\n")
            file.write("\n".join(["\t".join([key, str(value)]) for key, value in dists.items()]))
            file.write("\n\n")


if __name__ == "__main__":
    # cache_name = r"..\association_meter\cache"
    # with open(cache_name, "r") as file:
    #     data = json.load(file)

    # model = Word2Vec(data, min_count=1, vector_size=100, window=5, sg=0)
    # # min_count=1 - учитываются все слова, даже если они только один раз встречаются в тексте
    # # vector_size=100 - каждое слово будет представлено в виде вектора размерностью 100
    # # window=5 - размер окна равен 5
    # # sg=0 - использовать CBOW
    #
    # model.save("w2v_model")
    model = Word2Vec.load("w2v_model")

    ## Task 4
    cosine_dict, vector_list = get_cosine_distance_groups(model)
    save_cosine_distance(cosine_dict)

    for i, vector_dict in enumerate(vector_list):
        pca = PCA(n_components=2)

        vector_dict_norm = pca.fit_transform(list(vector_dict.values()))

        plt.scatter(vector_dict_norm[:, 0], vector_dict_norm[:, 1])

        for idx, word in enumerate(vector_dict.keys()):
            plt.annotate(word, (vector_dict_norm[idx, 0], vector_dict_norm[idx, 1]))

    plt.savefig(f"pca_embeddings.png")

    ## Task 7
