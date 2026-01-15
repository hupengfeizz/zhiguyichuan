n = 6


print(
    [
        True
        for num in str(n)
        if any(
            False if int(num) % i == 0 else True for i in range(2, int(int(n**0.5) + 1))
        )
    ]
)
