import random


def sample_excluding(n, x, a):
    if x == -1:
        return [num for num in range(1, n + 1) if num != a]

    if x > n - 1:
        raise ValueError("Cannot sample more elements than available excluding 'a'")

    sampled = random.sample(range(1, n), x)

    return [num if num < a else num + 1 for num in sampled]