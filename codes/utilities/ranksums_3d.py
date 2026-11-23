import numpy as np
from scipy.stats import rankdata
from scipy.stats import norm

def ranksums_3d(x, y):
    """
    https://github.com/scipy/scipy/blob/v1.6.0/scipy/stats/stats.py#L7191-L7249
    Compute the Wilcoxon rank-sum statistic for two samples.
    The Wilcoxon rank-sum test tests the null hypothesis that two sets
    of measurements are drawn from the same distribution.  The alternative
    hypothesis is that values in one sample are more likely to be
    larger than the values in the other sample.
    This test should be used to compare two samples from continuous
    distributions.  It does not handle ties between measurements
    in x and y.  For tie-handling and an optional continuity correction
    see `scipy.stats.mannwhitneyu`.
    Parameters
    ----------
    x,y : array_like
        The data from the two samples.
    Returns
    -------
    statistic : float
        The test statistic under the large-sample approximation that the
        rank sum statistic is normally distributed.
    pvalue : float
        The two-sided p-value of the test.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Wilcoxon_rank-sum_test
    Examples
    --------
    We can test the hypothesis that two independent unequal-sized samples are
    drawn from the same distribution with computing the Wilcoxon rank-sum
    statistic.
    >>> from scipy.stats import ranksums
    >>> sample1 = np.random.uniform(-1, 1, 200)
    >>> sample2 = np.random.uniform(-0.5, 1.5, 300) # a shifted distribution
    >>> ranksums(sample1, sample2)
    RanksumsResult(statistic=-7.887059, pvalue=3.09390448e-15)  # may vary
    The p-value of less than ``0.05`` indicates that this test rejects the
    hypothesis at the 5% significance level.
    """
    #x, y = map(np.asarray, (x, y))
    # n1 = len(x)
    # n2 = len(y)
    # alldata = np.concatenate((x, y))
    # ranked = rankdata(alldata,axis = 0)
    # x = ranked[:n1]
    # s = np.sum(x, axis=0)
    # expected = n1 * (n1+n2+1) / 2.0
    # z = (s - expected) / np.sqrt(n1*n2*(n1+n2+1)/12.0)
    # prob = 2 * norm.sf(abs(z))
    # return z,prob

    n1 = np.count_nonzero(~np.isnan(x),axis=0)
    n2 = np.count_nonzero(~np.isnan(y),axis=0)
    alldata = np.concatenate((x, y))
    #ranked = mstats.rankdata(np.ma.masked_invalid(alldata),axis = 0)
    ranked = rankdata(np.ma.masked_invalid(alldata),axis =0)
    ranked[np.isnan(alldata)] = np.nan
    x = ranked[:len(x)]
    s = np.nansum(x, axis=0)
    expected = n1 * (n1+n2+1) / 2.0
    z = (s - expected) / np.sqrt(n1*n2*(n1+n2+1)/12.0)
    prob = 2 * norm.sf(abs(z))
    return z,prob