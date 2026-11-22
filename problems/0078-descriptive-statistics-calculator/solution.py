import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    data = np.asarray(data, dtype=float)
    mean = np.mean(data)
    median = np.median(data)
    var = np.var(data)
    std = np.sqrt(var)

    vals, cnts = np.unique(data, return_counts=True)
    mode = vals[np.argmax(cnts)]

    p25, p50, p75 = np.percentile(data, [25, 50, 75])
    iqr = p75 - p25
    
    return {
        "mean": mean,
        "median": median,
        "mode": mode,
        "variance": np.round(var, 4),
        "standard_deviation": np.round(std, 4),
        "25th_percentile": p25,
        "50th_percentile": p50,
        "75th_percentile": p75,
        "interquartile_range": iqr,
    }
