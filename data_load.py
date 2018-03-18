from os import listdir

from os.path import join
from sklearn.model_selection import train_test_split


def load_data():
    files = listdir('../images')
    files = [join('../images', x) for x in files]
    train, valid_test = train_test_split(files, test_size=0.1)
    valid, test = train_test_split(valid_test, test_size=0.5)
    return train, valid, test
