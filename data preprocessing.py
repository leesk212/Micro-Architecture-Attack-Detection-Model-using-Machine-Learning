import numpy as np
import pandas as pd
import tabulate
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import preprocessing

test = pd.read_csv('./train_concetenate_v2.csv')

pd.set_option('display.float_format', '{:.2f}'.format)  # 항상 float 형식으로


print(test.info)
print(test.to_markdown())
print(test.columns)
column_list = list(test.columns.values)
# colum_label_elemnet = enumerate(column_list)
index_list = list(test.index.values)
plt_list = []
plt_x = np.arange(0, 25, 0.1)


def plot(plt_each_row):
    Index_name = plt_each_row[0]
    plt_y_0 = plt_each_row[1]
    plt_y_1 = plt_each_row[2]
    plt_y_2 = plt_each_row[3]
    plt_y_3 = plt_each_row[4]
    plt.xlabel('Time(0.1s)')
    plt.ylabel(Index_name)
    plt.plot(plt_x, plt_y_0, color='r', label='Normal')
    plt.plot(plt_x, plt_y_1, label='Flush+Reload')
    plt.plot(plt_x, plt_y_2, label='Flush+Flush')
    plt.plot(plt_x, plt_y_3, label='Meltdown')
    plt.grid()
    plt.legend()
    plt.show()


def make_state_list(state, each_column_label):
    count = 0
    value = []
    for index, row in test.iterrows():
        if int(row['State']) == state:
            count = count + 1
            value.append(row[each_column_label])
            if count == 250:
                return value


if __name__ == '__main__':
    del column_list[0]
    for each_column_label in column_list:
        new_list = [each_column_label, make_state_list(0, each_column_label), make_state_list(1, each_column_label),
                    make_state_list(2, each_column_label), make_state_list(3, each_column_label)]
        plt_list.append(new_list)

    for each_state in plt_list:
        plot(each_state)
