import math

def get_code_blocks(code):
    # Todo divide code into lines
    return []

def is_code_duplicate(code1, code2, threshold=0.65):
    # Todo calculate similarity ratio of code1 and code2
    return False

def longest_common_prefix(code):
    # Todo return the last lines of code share the same prefix
    return ""

def fix_repeats(code, threshold=0.65):
    lines = get_code_blocks(code)
    length = len(lines)

    # 记录重复行
    duplicate_lines = [False] * length

    # 遍历代码块大小和起始位置，检测连续代码块是否重复
    for block_size in range(1, math.floor(length / 2) + 1):
        i = 0
        max_length = 0
        cur_length = 0
        while i < length - block_size:
            actual_length = len(lines[i + block_size:i + 2 * block_size])
            code1 = "\n".join(lines[i:i + actual_length])
            code2 = "\n".join(lines[i + block_size:i + 2 * block_size])
            code1 = code1[:len(code2)] if actual_length < block_size else code1

            # 将重复行置为True
            if is_code_duplicate(code1, code2, 0.8):
                duplicate_lines[i + block_size:i + 2 * block_size] = [True] * block_size
                cur_length += actual_length
                if not duplicate_lines[i]:
                    cur_length += block_size
                i += block_size  # Skip over this block
            else:
                max_length = max(max_length, cur_length)
                cur_length = 0
                i += 1
        max_length = max(max_length, cur_length)
        # 判断重复行与总行数比例超过阈值
        if max_length / length > threshold:
            break
        else:
            duplicate_lines = [False] * length
    # 去除重复行，保留第一块
    fixed_lines = [line for i, line in enumerate(lines) if not duplicate_lines[i]]
    return "\n".join(fixed_lines)


def delete_repeat_function_Python(code):
    lines = [line for line in code.split("\n") if line.strip()]
    if len(lines) == 0:
        return ""
    res = [lines[0]]
    for line in lines[1:]:
        if line.find(line.strip()) == 0:
            break
        res.append(line)
    return ("\n" if code.startswith("\n") else "") + "\n".join(res)


def fix_repeats_in_line(line, sp_tokens, language="Python"):
    if len(line["output"].split("\n")[-1]) > 100:
        line["output"] = line["output"][:line["output"].rfind("\n")]

    # 去掉前缀
    prefix_len = len(line["input"])
    if sp_tokens[0]:
        prefix_len += len(sp_tokens[0])
    code = line["output"][prefix_len:]
    # 去掉后缀
    suffix_start = code.find(sp_tokens[1])
    if suffix_start >= 0:
        code = code[:code.find(sp_tokens[1])]

    repeat_code = longest_common_prefix(code)
    if repeat_code:
        n = repeat_code.count("\n")
        lines = code.split("\n")
        code = '\n'.join(lines[:-n+2])
        # return line
    if language == "Python":
        code = delete_repeat_function_Python(code)

    fixed_code = fix_repeats(code)
    fixed_code = sp_tokens[0] + line["input"] + fixed_code
    line["output"] = fixed_code

    return line