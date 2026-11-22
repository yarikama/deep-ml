def dice_statistics(n: int) -> tuple[float, float]:
    """
    Compute the expected value and variance of a fair n-sided die roll.

    Args:
        n (int): Number of sides of the die

    Returns:
        tuple: (expected_value, variance)
    """
    # Your code here
    e = (n + 1) / 2
    var = sum((i - e) ** 2 for i in range(1, n+1)) / n
    return round(e, 4), round(var, 4) 