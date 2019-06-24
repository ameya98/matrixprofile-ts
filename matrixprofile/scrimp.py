# -*- coding: utf-8 -*-

"""
This module consists of all code to implement the SCRIMP++ algorithm. SCRIMP++
is an anytime algorithm that computes the matrix profile for a given time
series (ts) over a given window size (m).

This algorithm was originally created at the University of California
Riverside. For further academic understanding, please review this paper:

Matrix Proﬁle XI: SCRIMP++: Time Series Motif Discovery at Interactive
Speed. Yan Zhu, Chin-Chia Michael Yeh, Zachary Zimmerman, Kaveh Kamgar
Eamonn Keogh, ICDM 2018.

https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

range = getattr(__builtins__, 'xrange', range)
# end of py2 compatability boilerplate

import math
import time

import warnings

import numpy as np


def fast_find_nn_pre(ts, m):
    n = len(ts)
    X = np.fft.fft(ts)
    cum_sumx = np.cumsum(ts)
    cum_sumx2 = np.cumsum(np.power(ts, 2))
    sumx = cum_sumx[m - 1:n] - np.insert(cum_sumx[0:n - m], 0, 0)
    sumx2 = cum_sumx2[m - 1:n] - np.insert(cum_sumx2[0:n - m], 0, 0)
    meanx = sumx / m
    sigmax2 = (sumx2 / m) - np.power(meanx, 2)
    sigmax = np.sqrt(sigmax2)

    return X, n, sumx2, sumx, meanx, sigmax2, sigmax


def calc_distance_profile(X, y, n, m, meanx, sigmax, std_noise=0):
    # reverse the query
    y = np.flip(y, 0)

    # make y same size as ts with zero fill
    y = np.concatenate([y, np.zeros(n - m)])

    # main trick of getting dot product in O(n log n) time
    Y = np.fft.fft(y)
    Z = X * Y
    z = np.fft.ifft(Z)

    # compute y stats in O(n)
    sumy = np.sum(y)
    sumy2 = np.sum(np.power(y, 2))
    meany = sumy / m
    sigmay2 = sumy2 / m - meany ** 2
    sigmay = np.sqrt(sigmay2)

    dist = (z[m - 1:n] - m * meanx * meany) / (sigmax * sigmay)
    dist = m - dist
    dist = 2 * dist

    # Noise correction.
    dist -= (2 + 2 * m) * np.square(std_noise / np.maximum(sigmay, sigmax))

    # Correct negative values.
    dist[dist < 0] = 0

    return np.real(np.sqrt(dist))


def apply_exclusion_zone(idx, exclusion_zone, profile_len, distance_profile):
    exc_start = np.maximum(0, idx - exclusion_zone)
    exc_stop = np.minimum(profile_len, idx + exclusion_zone)
    distance_profile[exc_start:exc_stop] = np.inf
    return distance_profile


def find_and_store_nn(iteration, idx, matrix_profile, mp_index,
                      distance_profile):
    if iteration == 0:
        matrix_profile = distance_profile
        mp_index[:] = idx
    else:
        update_pos = distance_profile < matrix_profile
        mp_index[update_pos] = idx
        matrix_profile[update_pos] = distance_profile[update_pos]

    idx_min = np.argmin(distance_profile)
    matrix_profile[idx] = distance_profile[idx_min]
    mp_index[idx] = idx_min
    idx_nn = mp_index[idx]

    return matrix_profile, mp_index, idx_nn


def calc_idx_diff(idx, idx_nn):
    return idx_nn - idx


def calc_dotproduct_idx(dotproduct, m, mp, idx, sigmax, idx_nn, meanx):
    dotproduct[idx] = (m - mp[idx] ** 2 / 2) * sigmax[idx] * sigmax[idx_nn] + m * meanx[idx] * meanx[idx_nn]

    return dotproduct


def calc_end_idx(profile_len, idx, step_size, idx_diff):
    return np.min([profile_len - 1, idx + step_size - 1, profile_len - idx_diff - 1])


def calc_dotproduct_end_idx(ts, dp, idx, m, endidx, idx_nn, idx_diff):
    tmp_a = ts[idx + m:endidx + m]
    tmp_b = ts[idx_nn + m:endidx + m + idx_diff]
    tmp_c = ts[idx:endidx]
    tmp_d = ts[idx_nn:endidx + idx_diff]
    tmp_f = tmp_a * tmp_b - tmp_c * tmp_d

    dp[idx + 1:endidx + 1] = dp[idx] + np.cumsum(tmp_f)

    return dp


def calc_refine_distance_end_idx(refine_distance, dp, idx, endidx, meanx,
                                 sigmax, idx_nn, idx_diff, m, std_noise=0):
    tmp_a = dp[idx + 1:endidx + 1]
    tmp_b = meanx[idx + 1:endidx + 1]
    tmp_c = meanx[idx_nn + 1:endidx + idx_diff + 1]
    tmp_d = sigmax[idx + 1:endidx + 1]
    tmp_e = sigmax[idx_nn + 1:endidx + idx_diff + 1]
    tmp_f = tmp_b * tmp_c
    tmp_g = tmp_d * tmp_e
    tmp_h = m - (tmp_a - m * tmp_f) / tmp_g
    tmp_h *= 2

    dist = tmp_h

    # Noise correction.
    dist -= (2 + 2 * m) * np.square(std_noise / np.maximum(tmp_d, tmp_e))

    # Correct negative values.
    dist[dist < 0] = 0

    refine_distance[idx + 1:endidx + 1] = np.sqrt(dist)

    return refine_distance


def calc_begin_idx(idx, step_size, idx_diff):
    return np.max([0, idx - step_size + 1, 2 - idx_diff])


def calc_dotproduct_begin_idx(ts, dp, beginidx, idx, idx_diff, m,
                              idx_nn):
    indices = np.arange(idx - 1, beginidx - 1, -1)
    tmp_a = ts[indices]
    tmp_b = ts[np.arange(idx_nn - 1, beginidx + idx_diff - 1, -1)]
    tmp_c = ts[np.arange(idx + m - 1, beginidx + m - 1, -1)]
    tmp_d = ts[np.arange(idx_nn - 1 + m, beginidx + idx_diff + m - 1, -1)]

    dp[indices] = dp[idx] + np.cumsum((tmp_a * tmp_b) - (tmp_c * tmp_d))

    return dp


def calc_refine_distance_begin_idx(refine_distance, dp, beginidx, idx,
                                   idx_diff, idx_nn, sigmax, meanx, m, std_noise=0):
    if beginidx >= idx:
        return refine_distance

    tmp_a = dp[beginidx:idx]
    tmp_b = meanx[beginidx:idx]
    tmp_c = meanx[beginidx + idx_diff:idx_nn]
    tmp_d = sigmax[beginidx:idx]
    tmp_e = sigmax[beginidx + idx_diff:idx_nn]
    tmp_f = tmp_b * tmp_c
    tmp_g = tmp_d * tmp_e
    tmp_h = m - (tmp_a - m * tmp_f) / tmp_g
    tmp_h *= 2

    dist = tmp_h

    # Noise correction.
    dist -= (2 + 2 * m) * np.square(std_noise / np.maximum(tmp_d, tmp_e))

    # Correct negative values.
    dist[dist < 0] = 0

    refine_distance[beginidx:idx] = np.sqrt(dist)

    return refine_distance


def apply_update_positions(matrix_profile, mp_index, refine_distance, beginidx,
                           endidx, idx_diff):
    # Following the paper's notation,
    # d == refine_distance[beginidx: endidx + 1]
    # j - i == idx_diff

    update_pos1 = np.where(refine_distance[beginidx: endidx + 1] < matrix_profile[beginidx: endidx + 1])[0]
    update_pos1 += beginidx
    matrix_profile[update_pos1] = refine_distance[update_pos1]
    mp_index[update_pos1] = update_pos1 + idx_diff

    update_pos2 = np.where(refine_distance[beginidx: endidx + 1] < matrix_profile[beginidx + idx_diff: endidx + idx_diff + 1])[0]
    update_pos2 += beginidx
    matrix_profile[update_pos2 + idx_diff] = refine_distance[update_pos2]
    mp_index[update_pos2 + idx_diff] = update_pos2

    return matrix_profile, mp_index


def calc_curlastz(ts, m, n, idx, profile_len, curlastz):
    curlastz[idx] = np.sum(ts[0:m] * ts[idx:idx + m])

    tmp_a = ts[m:n - idx]
    tmp_b = ts[idx + m:n]
    tmp_c = ts[0:profile_len - idx - 1]
    tmp_d = ts[idx:profile_len - 1]
    tmp_e = tmp_a * tmp_b
    tmp_f = tmp_c * tmp_d
    curlastz[idx + 1:profile_len] = curlastz[idx] + np.cumsum(tmp_e - tmp_f)

    return curlastz


def calc_curdistance(curlastz, meanx, sigmax, idx, profile_len, m,
                     curdistance, std_noise=0):
    tmp_a = curlastz[idx:profile_len + 1]
    tmp_b = meanx[idx:profile_len]
    tmp_c = meanx[0:profile_len - idx]
    tmp_d = sigmax[idx:profile_len]
    tmp_e = sigmax[0:profile_len - idx]
    tmp_f = tmp_b * tmp_c
    tmp_g = m - (tmp_a - m * tmp_f) / (tmp_d * tmp_e)
    tmp_g *= 2
    dist = tmp_g

    # Noise correction.
    dist -= (2 + 2 * m) * np.square(std_noise / np.maximum(tmp_d, tmp_e))

    # Correct negative values.
    dist[dist < 0] = 0

    curdistance[idx:profile_len] = np.sqrt(dist)

    return curdistance


def time_is_exceeded(start_time, runtime):
    """Helper method to determine if the runtime has exceeded or not.

    Returns
    -------
    bool
        Whether or not the runtime has exceeded.
    """
    elapsed = time.time() - start_time
    exceeded = runtime is not None and elapsed >= runtime
    if exceeded:
        warnings.warn(
            'Max runtime exceeded. Approximate solution is given.',
            RuntimeWarning
        )

    return exceeded


def scrimp_plus_plus(ts, window_size, step_size_fraction=0.25, runtime=None, random_state=None, std_noise=0,
                     exclusion_zone_fraction=0.25):
    """
    SCRIMP++ is an anytime algorithm that computes the matrix profile for a
    given time series (ts) over a given window size (m). Essentially, it allows
    for an approximate solution to be provided for quicker analysis. In the
    case of this implementation, the runtime is measured based on the wall
    clock. If the number of seconds exceeds the runtime, then the approximate
    solution is returned. If the runtime is None, the exact solution is
    returned.

    This algorithm was created at the University of California Riverside. For
    further academic understanding, please review this paper:

    Matrix Proﬁle XI: SCRIMP++: Time Series Motif Discovery at Interactive
    Speed. Yan Zhu, Chin-Chia Michael Yeh, Zachary Zimmerman, Kaveh Kamgar
    Eamonn Keogh, ICDM 2018.

    https://www.cs.ucr.edu/~eamonn/SCRIMP_ICDM_camera_ready_updated.pdf

        Parameters
        ----------
        ts : np.ndarray
            The time series to compute the matrix profile for.
        window_size : int
            The window size.
        step_size_fraction : float, default 0.25
            Fraction that decides the sampling interval for the window. The paper suggests 0.25 is the
            most practical.
        runtime : int, default None
            The maximum number of seconds based on wall clock time for this
            algorithm to run. It computes the exact solution when it is set to
            None.
        random_state : int, default None
            Set the random seed generator for reproducible results.
        std_noise: float, default 0
            Noise standard deviation for noise correction.
        exclusion_zone_fraction: float, default 0.25
            Fraction of the window size deciding the exclusion zone for trivial matches.

        Returns
        -------
        (np.array, np.array)
            The matrix profile and the matrix profile index respectively.
    """
    # Start the timer here.
    start_time = time.time()

    # Validate step_size.
    if step_size_fraction < 0:
        raise ValueError('Parameter step_size should be non-negative.')

    # Validate random_state.
    if random_state is not None:
        try:
            np.random.seed(random_state)
        except ValueError:
            raise ValueError('Invalid random_state value given.')

    ts_len = len(ts)

    # Validate window size.
    if window_size > ts_len / 2:
        raise ValueError('Time series is too short relative to desired subsequence length.')

    if window_size < 4:
        raise ValueError('Window size must be at least 4.')

    # Set the trivial match range.
    exclusion_zone = np.floor(exclusion_zone_fraction * window_size).astype(int)

    # Initialization.
    step_size = np.ceil(window_size * step_size_fraction).astype(int)

    profile_len = ts_len - window_size + 1
    matrix_profile = np.zeros(profile_len)
    mp_index = np.zeros(profile_len).astype(int)

    X, n, sumx2, sumx, meanx, sigmax2, sigmax = fast_find_nn_pre(ts, window_size)

    # PreSCRIMP.
    dotproduct = np.zeros(profile_len)
    refine_distance = np.full(profile_len, np.inf)
    compute_order = np.arange(0, profile_len, step_size).astype(int)
    np.random.shuffle(compute_order)

    for iteration, idx in enumerate(compute_order):
        # Compute distance profile.
        subsequence = ts[idx: idx + window_size]

        distance_profile = calc_distance_profile(X, subsequence, n, window_size, meanx,
                                                 sigmax, std_noise=std_noise)

        # Apply exclusion zone.
        distance_profile = apply_exclusion_zone(idx, exclusion_zone, profile_len, distance_profile)

        # Find and store nearest neighbor.
        matrix_profile, mp_index, idx_nn = find_and_store_nn(
            iteration, idx, matrix_profile, mp_index, distance_profile)

        # Compute distances between queries starting close to this index, and queries starting close to the nearest neighbour.
        idx_diff = calc_idx_diff(idx, idx_nn)
        dotproduct = calc_dotproduct_idx(dotproduct, window_size, matrix_profile, idx,
                                         sigmax, idx_nn, meanx)

        endidx = calc_end_idx(profile_len, idx, step_size, idx_diff)

        dotproduct = calc_dotproduct_end_idx(ts, dotproduct, idx, window_size,
                                             endidx, idx_nn, idx_diff)

        refine_distance = calc_refine_distance_end_idx(refine_distance, dotproduct, idx, endidx, meanx, sigmax, idx_nn,
                                                       idx_diff, window_size, std_noise=std_noise)

        beginidx = calc_begin_idx(idx, step_size, idx_diff)

        dotproduct = calc_dotproduct_begin_idx(ts, dotproduct, beginidx, idx, idx_diff, window_size, idx_nn)

        refine_distance = calc_refine_distance_begin_idx(refine_distance, dotproduct, beginidx, idx, idx_diff, idx_nn,
                                                         sigmax, meanx, window_size, std_noise=std_noise)

        # Update matrix profile if we can.
        matrix_profile, mp_index = apply_update_positions(matrix_profile, mp_index,
                                                          refine_distance, beginidx, endidx, idx_diff)

        # Stop if time is up.
        if time_is_exceeded(start_time, runtime):
            return matrix_profile, mp_index

    # SCRIMP.
    compute_order = np.arange(exclusion_zone + 1, profile_len)
    np.random.shuffle(compute_order)

    curlastz = np.zeros(profile_len)
    curdistance = np.zeros(profile_len)
    dist1 = np.full(profile_len, np.inf)
    dist2 = np.full(profile_len, np.inf)

    for idx in compute_order:
        curlastz = calc_curlastz(ts, window_size, n, idx, profile_len, curlastz)
        curdistance = calc_curdistance(curlastz, meanx, sigmax, idx,
                                       profile_len, window_size, curdistance, std_noise=std_noise)

        dist1[0: idx] = np.inf
        dist1[idx:profile_len] = curdistance[idx:profile_len]

        dist2[0: profile_len - idx] = curdistance[idx:profile_len]
        dist2[profile_len - idx:profile_len] = np.inf

        loc1 = dist1 < matrix_profile
        matrix_profile[loc1] = dist1[loc1]
        mp_index[loc1] = np.arange(profile_len)[loc1] - idx + 1

        loc2 = dist2 < matrix_profile
        matrix_profile[loc2] = dist2[loc2]
        mp_index[loc2] = np.arange(profile_len)[loc2] + idx - 1

        # Stop if time is up.
        if time_is_exceeded(start_time, runtime):
            return matrix_profile, mp_index

    return matrix_profile, mp_index
