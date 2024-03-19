import math
import os
from typing import TextIO

punctuation_marks = ['.', ',', '!', ':', '?', '/', ')', '(', '@', '$', '#', '%', '^']


def replace_all(data, to_replace: list, new_value):
    for element_to_replace in to_replace:
        data = data.replace(element_to_replace, new_value)

    return data


def normalize_file(file: TextIO, stop_words: list):
    contents = file.read()

    contents = contents.lower()
    contents = replace_all(contents, punctuation_marks, '')
    contents = replace_all(contents, ['\n', '\r'], ' ')
    contents = replace_all(contents, ['  '], ' ')

    contents = contents.split(' ')
    contents = [content for content in contents if content not in stop_words]

    return contents


def read_folder(path: str, input_files: list, stop_words: list):
    files = os.listdir(path)
    word_occurences = {}
    total_words = 0
    total_files = 0

    for (i, file) in enumerate(files):
        file_path = os.path.join(path, file)

        if file not in input_files:
            continue

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            normalized_content = normalize_file(f, stop_words)

            for content in normalized_content:
                if content in word_occurences:
                    word_occurences[content] += 1
                else:
                    word_occurences[content] = 1

                total_words += 1

            total_files += 1

    return word_occurences, total_files, total_words


def read_file(file_name: str):
    words = []

    with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            parsed_line = replace_all(line, ['\n', '\r'], '')

            words.append(parsed_line)

    return words


def read_data_file_names(file_name: str):
    spam_files = []
    ham_files = []

    with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            line = replace_all(line, ['\n', '\r'], '')

            if 'spam' in line:
                spam_files.append(line)
            elif 'ham' in line:
                ham_files.append(line)

    return spam_files, ham_files


def divide_all(dictionary: dict, divide_with: int, additive_constant: float, full_dictionary_len: int):
    new_dict = {}

    for key in dictionary:
        new_dict[key] = (dictionary[key] + additive_constant) / (divide_with + additive_constant * full_dictionary_len)

    return new_dict


def conditional_get(dictionary: dict, key: str, default_value: float = 10e-9):
    if key not in dictionary:
        return default_value

    if dictionary[key] < default_value:
        return default_value

    return dictionary[key]


def test_for(files, p_wk_spam: dict, p_wk_ham: dict, p_spam: float, p_ham: float, stop_words: list):
    total_count = 0
    correct_count = 0
    false_positive_count = 0
    false_negative_count = 0
    total_ham_count = 0
    total_spam_count = 0

    for file in files:
        test_type = 'spam'

        if 'ham' in file:
            test_type = 'ham'

        file_path = os.path.join(test_type, file)

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            normalized_content = normalize_file(f, stop_words)
            ln_r = math.log(p_spam) - math.log(p_ham)

            document_word_occurrences = {}

            for word in normalized_content:
                if word in document_word_occurrences:
                    document_word_occurrences[word] += 1
                else:
                    document_word_occurrences[word] = 1

            for word in normalized_content:
                word_spam_possibility = conditional_get(p_wk_spam, word)
                word_ham_possibility = conditional_get(p_wk_ham, word)

                word_document_occurrence = conditional_get(document_word_occurrences, word, 0)

                second_part = math.log(word_spam_possibility) - math.log(word_ham_possibility)
                ln_r += word_document_occurrence * second_part

            if test_type == 'spam' and ln_r > 0:
                correct_count += 1
            elif test_type == 'spam' and ln_r <= 0:
                false_negative_count += 1

            if test_type == 'ham' and ln_r <= 0:
                correct_count += 1
            elif test_type == 'ham' and ln_r > 0:
                false_positive_count += 1

            if test_type == 'ham':
                total_ham_count += 1
            else:
                total_spam_count += 1

        total_count += 1

    accuracy = correct_count / total_count

    false_negative_rate = float('nan')
    false_positive_rate = float('nan')

    if total_ham_count != 0:
        false_positive_rate = false_positive_count / total_ham_count

    if total_spam_count != 0:
        false_negative_rate = false_negative_count / total_spam_count

    return accuracy, false_positive_rate, false_negative_rate


def test_stats(file_name: str, p_wk_spam, p_wk_ham, p_spam, p_ham, stop_words):
    test_spam_files, test_ham_files = read_data_file_names(file_name)
    accuracy, false_positive_rate, false_negative_rate = test_for(test_spam_files + test_ham_files, p_wk_spam, p_wk_ham, p_spam, p_ham,
                                                                  stop_words)

    error = 1 - accuracy

    print(f'Helyesseg: {accuracy * 100}%')

    print(f'Spam false positive hibaarany: {false_positive_rate * 100}%')
    print(f'Spam false negative hibaarany: {false_negative_rate * 100}%')

    print(f'Hiba: {error * 100}%')


def binary_classification(spam_data, ham_data, stop_words, additive_constant: float):
    print('----------------------------------------------')
    print(f'\nAdditiv simitas, alfa: {additive_constant}')

    spam_words, total_spam_files, total_spam_words = spam_data
    ham_words, total_ham_files, total_ham_words = ham_data
    total_dictionary_len = spam_words + ham_words

    total_files = total_spam_files + total_ham_files

    p_spam = total_spam_files / total_files
    p_ham = total_ham_files / total_files

    p_wk_spam = divide_all(spam_words, total_spam_words, additive_constant, total_dictionary_len)
    p_wk_ham = divide_all(ham_words, total_ham_words, additive_constant, total_dictionary_len)

    print('\n--------------------------------------------------')
    print('\nStatisztika tanulasi adatokra: \n')

    test_stats('train.txt', p_wk_spam, p_wk_ham, p_spam, p_ham, stop_words)

    print('\n--------------------------------------------------')
    print('Statisztika teszt adatokra: \n')

    test_stats('test.txt', p_wk_spam, p_wk_ham, p_spam, p_ham, stop_words)

    print('\n\n\n')


def main():
    stop_words = read_file('stopwords2.txt')
    stop_words.append('subject')

    training_spam_files, training_ham_files = read_data_file_names('train.txt')

    spam_data = read_folder('spam', training_spam_files, stop_words)
    ham_data = read_folder('ham', training_ham_files, stop_words)

    binary_classification(spam_data, ham_data, stop_words, 0.01)
    binary_classification(spam_data, ham_data, stop_words, 0.1)
    binary_classification(spam_data, ham_data, stop_words, 1)


if __name__ == '__main__':
    main()
