import numpy as np
import scipy

class KS_2_sample_test:

    def __init__(self):
        self.null_hypothesis = "There is NO difference between Sample 1 (array_a) and Sample 2 (array_b)"
        self.alternate_hypothesis = "There is a difference between Sample 1 (array_a) and Sample 2 (array_b)"

    def _concatenate_obs(self, array_a, array_b):
        if str(type(array_a)) != "<class 'numpy.ndarray'>" or str(type(array_b)) != "<class 'numpy.ndarray'>":
            raise ValueError('All samples must be numpy arrays')
        return np.array(
            sorted(
                np.concatenate((array_a, array_b))
            )
        )

    def _sort_samples(self, array_a, array_b):
        return (
            np.array(
                sorted(array_a)
            ), 
            np.array(
                sorted(array_b)
            )
        )

    def _compute_cdf(self, sorted_array, concatenated_observations):
        result = []
        for x in concatenated_observations:
            cdf = len(
                sorted_array[np.where(sorted_array <= x)]
            )/len(sorted_array)
            result.append(cdf)
        return np.array(result)

    def _compute_ks_statistic(self, cdf_a, cdf_b):
        return max(
            abs(
                cdf_a - cdf_b
            )
        )
        
    def __compute_en(self, array_a, array_b):
        m = len(array_a)
        n = len(array_b)
        en = m * n/(m + n)
        return en

    def _compute_p_value(self, k_stat, en):
        p_value = scipy.stats.kstwo.sf(k_stat, np.round(en))
        return p_value

    def run_test(self, array_a, array_b, alpha):
        sorted_array_a, sorted_array_b = self._sort_samples(array_a, array_b)
        concatenated_array = self._concatenate_obs(array_a, array_b)
        cdf_a = self._compute_cdf(sorted_array_a, concatenated_array)
        cdf_b = self._compute_cdf(sorted_array_b, concatenated_array)
        ks_stat = self._compute_ks_statistic(cdf_a, cdf_b)
        en = self.__compute_en(array_a, array_b)
        p_value = self._compute_p_value(ks_stat, en)
        print("Null Hypothesis:", self.null_hypothesis)
        print("Alternate Hypothesis:", self.alternate_hypothesis)
        print('\n')
        print(
            {
            'ks-statistic':ks_stat,
            'p-value':p_value
            }
        )
        print('\n')
        if p_value < alpha:
            print(f"Null Hypothesis Rejected, Reason: {p_value} < {alpha}")
        else:
            print(f"Null Hypothesis Accepted, Reason: {p_value} > {alpha}")
