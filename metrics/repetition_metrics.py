from collections import Counter

import numpy as np
import textdistance

from util.tokenizer import tokenize_code


def generate_ngrams(tokens, n):
    # 生成n-gram
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    return ngrams


def rep_n(code, n):
    # 使用分词函数获取标记
    tokens = tokenize_code(code)

    # 生成n-gram
    ngrams = generate_ngrams(tokens, n)

    # 计算n-gram的总数量
    total_ngrams = len(ngrams)

    # 使用Counter计算每个n-gram的出现次数
    ngram_counts = Counter(ngrams)

    # 计算独特n-gram的数量
    unique_ngrams = len(ngram_counts)

    # 计算rep_n
    rep_n_value = 100 * (1 - (unique_ngrams / total_ngrams) if total_ngrams > 0 else 0)

    return rep_n_value


def rep_line(code):
    # 按行分割代码
    lines = code.strip().split('\n')

    # 去除每行两端的空白字符
    lines = [line for line in lines if line.strip()]

    # 计算行的总数量
    total_lines = len(lines)

    # 使用Counter计算每行的出现次数
    line_counts = Counter(lines)

    # 计算独特行的数量
    unique_lines = len(line_counts)

    # 计算rep_line
    rep_line_value = 100 * (1 - (unique_lines / total_lines) if total_lines > 0 else 0)

    return rep_line_value


def sim_line(code, threshold=1):
    # 按行分割代码
    lines = code.strip().split('\n')

    lines = [tokenize_code(line) for line in lines if line.strip()]

    n = len(lines)

    if n <= 1:
        return 0

    # 初始化矩阵
    distance_matrix = np.zeros((n, n), dtype=int)

    # 计算Levenshtein距离并填充矩阵
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = textdistance.levenshtein(lines[i], lines[j])
            else:
                distance_matrix[i][j] = 0

    # 初始化一个列表来存储每一行的类编号
    line_classes = [-1] * n
    current_class = 0

    for i in range(n):
        if line_classes[i] == -1:  # 如果该行还没有被分类
            line_classes[i] = current_class
            for j in range(i + 1, n):
                # if line_classes[j] == -1 and lines[i] == lines[j]:
                if line_classes[j] == -1 and distance_matrix[i][j] <= threshold:
                    line_classes[j] = current_class
            current_class += 1

    # 计算相似行的类的数量
    unique_classes = len(set(line_classes))

    # 计算sim_line
    sim_line_value = 100 * (1 - (unique_classes / n))

    return sim_line_value

