/*! Some utilities for HMM
 * Author: Zixuan Wei
 * 
 * Classes:
 *     
 *     ConvergenceMonitor - Moniter the training process of the HMM
 * 
 */
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <Eigen/Dense>

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
};

/*!
 * \brief Log of sum of exp
 * \param X A sequence.
 * \return logsumexp
 */
inline double logsumexp(Eigen::RowVectorXd X, size_t size) {
  double max_value = 0;
  max_value = X.maxCoeff();
  if (std::isinf(max_value)) return -INFINITY;
  double acc = 0.0;
  for (size_t i = 0; i < size; i++) {
    acc += std::expl(X(i) - max_value);
  }

  return std::logl(acc + max_value);
}
inline double logsumexp(double *X, size_t size) {
  double max_value = *std::max_element(X, X + size);
  if (std::isinf(max_value)) return -INFINITY;
  double acc = 0.0;
  for (size_t i = 0; i < size; i++) {
    acc += std::expl(X[i] - max_value);
  }

  return std::logl(acc + max_value);
}

/*!
 * \brief Log of add of exp
 * \param a
 * \param b
 */
inline double logaddexp(double a, double b) {
  if (std::isinf(a) && a < 0) return b;
  else if (std::isinf(b) && b < 0) return a;
  else return std::max(a, b) + 
                std::log1pl(std::expl(-std::fabsl(a - b)));
}

void forward(int n_observations, int n_components,
             Eigen::VectorXd log_stateprob,
             Eigen::MatrixXd log_transmit,
             Eigen::MatrixXd framelogprob,
             Eigen::MatrixXd alpha);

void forward(int n_observations, int n_components,
             Eigen::VectorXd log_stateprob,
             Eigen::MatrixXd log_transmit,
             Eigen::MatrixXd framelogprob,
             Eigen::MatrixXd beta);

void compute_log_xi_sum(int n_observations, int n_components,
                        Eigen::MatrixXd alpha,
                        Eigen::MatrixXd log_transmit,
                        Eigen::MatrixXd bwdlattice,
                        Eigen::MatrixXd framelogprob,
                        Eigen::MatrixXd log_xi_sum);

void viterbi(int n_observations, int n_components,
             Eigen::VectorXd log_startprob,
             Eigen::MatrixXd log_transmit,
             Eigen::MatrixXd framelogprob,
             Eigen::VectorXd state_sequence,
             double *logprob);

/*!
 * \brief Iteration
 */
struct IterFromIndividualLength {
  size_t *start = nullptr;
  size_t *end = nullptr;
  size_t t = 0;
  size_t maxt = 0;
  size_t n_observations = 0;
  IterFromIndividualLength(const std::vector<double>& X, const std::vector<int>& lengths);
  size_t get_start();
  size_t get_end();
  ~IterFromIndividualLength();
};

template <typename T>
void alloc_mat(T **&a, size_t n_row, size_t n_col) {
  a = new T*[n_row];
  for (size_t i = 0; i < n_row; i++) {
    a[i] = new T[n_col];
  }
}

template <typename T>
void free_mat(T **&a, size_t n_row, size_t n_col) {
  for (size_t i = 0; i < n_row; i++) {
    delete [] a[i];
  }
  delete [] a;
}
