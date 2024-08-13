import re


def tokenize_code(code):
    # 定义一个正则表达式模式，用于匹配代码中的标识符、关键字、数字等
    pattern = re.compile(r'\b\w+\b')

    # 使用正则表达式找到所有匹配的部分
    tokens = pattern.findall(code)

    return tokens
