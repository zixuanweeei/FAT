/* Some utilities for HMM
 * Author: Zixuan Wei
 * 
 * Classes:
 *     
 *     ConvergenceMonitor - Moniter the training process of the HMM
 * 
 */
#include <vector>

struct ConvergenceMonitor {
  double tol;                  /* Convergence threshold. */
  int max_epoch;              /* Maximum number of epoch to update the patameters. */
  bool verbose;               /* If `true` then per-interaction convergence reports
                                are printed. Otherwise, the moniter is mute. */
  std::vector<double> history; /* The log probility of the data for the last two training
                                iteractions. */
  int iter;                   /* Iteration times. */
  ConvergenceMonitor(double tol = 1e-4, int max_epoch = 100, 
                     bool verbose = true);
  bool report(double logprob);
}