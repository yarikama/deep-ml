import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    """
    Compute derivative of composite functions using chain rule.
    
    Args:
        functions: List of function names (applied right to left)
                  Available: 'square', 'sin', 'exp', 'log'
        x: Point at which to evaluate derivative
    
    Returns:
        Derivative value at x
    
    Example:
        ['sin', 'square'] represents sin(x²)
        ['exp', 'sin', 'square'] represents exp(sin(x²))
    """
    # Your code here

    grad = 1.0

    for op in functions[::-1]:
        if op == "square":
            grad *= 2.0 * x
            x = x ** 2

        if op == "sin":
            grad *= np.cos(x)
            x = np.sin(x)

        if op == "exp":
            grad *= np.exp(x)
            x = np.exp(x)

        if op == "log":
            grad *= 1.0 / x
            x = np.log(x)

    return float(grad)