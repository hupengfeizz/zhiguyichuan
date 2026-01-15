import random
import string


def generate_random_strings(count=10, length=12):
    # 定义字符集：包含大写和小写字母
    letters = string.ascii_letters

    # 生成字符串列表
    result = [
        "".join(random.choice(letters) for _ in range(length)) for _ in range(count)
    ]
    return result


# 执行生成
strings = generate_random_strings(10, 12)

# 打印结果
for i, s in enumerate(strings, 1):
    print(f"{i}: {s}")
