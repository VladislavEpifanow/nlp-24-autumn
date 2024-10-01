import os
from tokenizer import tokenize_text

def read_file(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return text

def write_file(file_path, tokens):
    with open(file_path, 'w') as f:
        f.write(tokens)

def main():
    data_folder = '..\\data\\20news-bydate-test\\rec.sport.hockey'
    save_folder = '.\\projects\\danishevskiy-lab\\assets\\annotated-corpus\\hockey'
    os.makedirs(save_folder, exist_ok=True)
    filenames = os.listdir(data_folder)

    for filename in filenames:

        try:
            text = read_file(os.path.join(data_folder, filename))
        
            text = text.replace('>', '').replace('|', '').replace('\n', ' ').replace('-', '')
            token_annotation, _ = tokenize_text(text)
            write_file(os.path.join(save_folder, filename+'.tsv'), token_annotation)
        except Exception as e:
            print(f'Error for {filename}', e)

if __name__ == '__main__':
    main()