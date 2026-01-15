def gen(num):
    tri = []

    for i in range(num):
        row = [1] * (i + 1)

        for j in range(1, i):
            row[j] = tri[i - 1][j - 1] + tri[i - 1][j]

        tri.append(row)
    return tri


tri = gen(6)
for i in tri:
    print(i)
