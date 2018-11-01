#pragma once
#include <iostream>

/*
 * Hidden Markov Model with Gaussian emissions.
 * Author: Zixuan
 */
struct BaseHMM {
  int N;                          /* number of states; Q = {1, 2, 3, ..., N} */
  double **A;                     /* NxN matrix. A[i][j] is the transition prob 
                                    of going from state i at time t to state j at 
                                    time t + 1. */
  double means;                   /* means of Gaussian Emissions of each state; 
                                    miu = {miu1, miu2, miu3, ..., miuN} */
}