coins = [1, 2, 5]
amount = 14
coins.sort(reverse=True)
d = {}

while amount:
    for coin in coins:
        n, amount = divmod(amount, coin)
        # if amount == 0:
        if n != 0:
            d[coin] = n

print(d)
