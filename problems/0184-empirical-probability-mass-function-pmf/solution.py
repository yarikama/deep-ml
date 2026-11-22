from collections import Counter

def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    c = Counter(samples)
    total = len(samples)
    return [(val, cnt / total) for val, cnt in c.items()]
    