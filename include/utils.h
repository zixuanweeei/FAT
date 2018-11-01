/*! Some utilities for HMM
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

  /*!
   * \brief Constructor for ConvergenceMonitor class
   * \param tol Convergence threshold.
   * \param max_epoch Maximum number of epoch.
   * \param verbose Whether to print the convergence reports.
   */
  ConvergenceMonitor(double tol = 1e-4, int max_epoch = 100, 
                     bool verbose = true) 
    : tol(tol), max_epoch(max_epoch), verbose(verbose), iter(0) {};

  /*!
   * \brief Print the convergence reports and return a bool
            type indicating the process status.
   * \param logprob The current log probability of the data
   *        as computed by EM algorithm in the current iter-
   *        ation.
   * \return bool type Indicates the convergence status.
   */
  bool report(double logprob);
}