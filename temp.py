import numpy as np

n = 100
k = 4
S = np.random.randint(1, 100, n)
S.sort()


def mask_gen(shape: tuple):
    M = np.zeros(shape)
    ids = np.column_stack([np.arange(0, shape[0], 1), np.array([x % k for x in range(shape[0])])])
    for i in ids:
        M[tuple(i)] = 1
    return M.astype(bool)


matrix = mask_gen((n, k))


def res(m, k):
    l = [sum(S[m[:, i]]) for i in range(k)]
    print(l)
    return max(l) - min(l)


res(matrix, k)


def opt(m, n, max_iters=100):
    iters = 0
    indexes = np.array(list(range(1, n)))
    probs = indexes / sum(indexes)
    indexes = np.random.choice(indexes, n - 1, replace=False, p=probs)
    combs = mask_gen((k,k))
    for index in indexes:
        temp = m
        cur = 100000000

        for comb in combs:
            temp[index] = comb
            t = res(temp, k)
            if cur > t:
                cur = t
                best = comb
        temp[index] = best

        iters += 1
        if iters > max_iters or t == 0:
            break
        print(res(temp, k))
    return temp


result = opt(matrix, 100)
