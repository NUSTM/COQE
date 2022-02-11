
def get_coordinate(x, y):
    return "%x[" + str(x) + "," + str(y) + "]"


def create_template_6(write_path):
    write_str, number = "# Unigram\n", 0

    # 单个特征
    for i in range(0, 5):
        for j in range(-3, 4):
            write_str += "U" + str(number) + ":" + get_coordinate(j, i) + "\n"
            number += 1
        write_str += "\n"

    # 词 + 词性
    write_str += "U" + str(number) + ":" + get_coordinate(0, 0) + "/" + get_coordinate(0, 1) + "\n"
    number += 1

    # 两两交叉特征
    for i in range(1, 4):
        for j in range(i + 1, 5):
            write_str += "U" + str(number) + ":" + get_coordinate(0, i) + "/" + get_coordinate(0, j) + "\n"
            number += 1
    write_str += "\n"

    # 单列特征前后组合
    for i in range(0, 5):
        for j in range(0, 3):
            if j == 0:
                write_str += "U" + str(number) + ":" + get_coordinate(j - 1, i) + "/" + get_coordinate(j, i) + "\n"
                number += 1
            write_str += "U" + str(number) + ":" + get_coordinate(j, i) + "/" + get_coordinate(j + 1, i) + "\n"
            number += 1
        write_str += "\n"

    # 单列特征，三元组合
    for i in range(0, 5):
        for j in range(-1, 2):
            write_str += "U" + str(number) + ":" + get_coordinate(j - 1, i) + "/" + get_coordinate(j, i) + "/" + get_coordinate(j + 1, i) + "\n"
            number += 1
            # write_str += "U" + str(number) + ":" + get_coordinate(j, i) + "/" + get_coordinate(j + 1, i) + "\n"
            # number += 1
        write_str += "\n"

    write_str += "# Bigram\nB"

    with open(write_path, "w", encoding="utf-8") as f:
        f.write(write_str)


def create_template_1(write_path):
    write_str, number = "# Unigram\n", 0

    # 单个特征
    for i in range(0, 1):
        for j in range(-3, 4):
            write_str += "U" + str(number) + ":" + get_coordinate(j, i) + "\n"
            number += 1
        write_str += "\n"

    # 单列特征前后组合
    for i in range(0, 1):
        for j in range(0, 3):
            if j == 0:
                write_str += "U" + str(number) + ":" + get_coordinate(j - 1, i) + "/" + get_coordinate(j, i) + "\n"
                number += 1
            write_str += "U" + str(number) + ":" + get_coordinate(j, i) + "/" + get_coordinate(j + 1, i) + "\n"
            number += 1
        write_str += "\n"

    write_str += "# Bigram\nB"

    with open(write_path, "w", encoding="utf-8") as f:
        f.write(write_str)



