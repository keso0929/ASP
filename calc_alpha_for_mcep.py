#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calculate alpha for mel-cepstral analysis
by brute-forcing over different values from 0 to 1. 
"""

import numpy as np

SAMPFREQ = 16000 # Sampling frequency

def calc_alpha_for_mcep(sampfreq=24000):
    num_points = 1000
    alpha_start = 0
    alpha_step = 0.001
    alpha_end = 1
    min_dist = float("inf")
    best_alpha = 0.

    mel = melscale_vector(sampfreq, num_points)

    for alpha in [alpha_step*x for x in range(int((alpha_end - alpha_start)/alpha_step))]:
        dist = RMSD(mel, warp_vector(alpha, num_points))
        #print(dist, min_dist, best_alpha)
        if dist < min_dist:
            min_dist = dist
            best_alpha = alpha
    
    print("Best Alpha for Frequency {0}: {1}".format(sampfreq, best_alpha))

def melscale_vector(sampfreq, vector_length):
    step = (float(sampfreq) / 2.) / float(vector_length)
    melscale_vector = np.zeros((vector_length, ), dtype="float64")

    for i in range(vector_length):
        f = step * float(i)
        melscale = (1000. / np.log(2)) * np.log(1. + (f / 1000.))
        melscale_vector[i] = melscale

    melscale_vector /= np.max(melscale_vector)

    return melscale_vector

def RMSD(vector1, vector2):
    return np.sqrt(np.sum((vector1 - vector2)**2)/len(vector1))

def warp_vector(alpha, vector_length):
    step = np.pi / vector_length
    warping_vector = melscale_vector = np.zeros((vector_length, ), dtype="float64")
    
    for i in range(vector_length):
        omega = step * float(i)
        num = (1 - alpha**2) * np.sin(omega)
        den = (1 + alpha**2) * np.cos(omega) - 2*alpha
        warp_freq = np.arctan(num / den)

        if warp_freq < 0:
            warp_freq += np.pi

        warping_vector[i] = warp_freq

    warping_vector /= np.max(warping_vector)

    return warping_vector

if __name__ == "__main__":
    calc_alpha_for_mcep(SAMPFREQ)
