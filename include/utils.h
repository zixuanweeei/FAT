/*! Some utilities for HMM
 * Author: Zixuan Wei
 * 
 * Classes:
 *     
 *     ConvergenceMonitor - Moniter the training process of the HMM
 * 
 */
#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <Eigen/Dense>
#include <random>

struct ConvergenceMonitor {
  double tol;                  /* Convergence threshold. */
  size_t max_epoch;              /* Maximum number of epoch to update the patameters. */
  bool verbose;               /* If `true` then per-interaction convergence reports
                                are printed. Otherwise, the moniter is mute. */
  std::vector<double> history; /* The log probility of the data for the last two training
                                iteractions. */
  size_t iter;                   /* Iteration times. */

  /*!
   * \brief Constructor for ConvergenceMonitor class
   * \param tol Convergence threshold.
   * \param max_epoch Maximum number of epoch.
   * \param verbose Whether to print the convergence reports.
   */
  ConvergenceMonitor(double tol = 1e-4, size_t max_epoch = 100, 
                     bool verbose = true)
    : tol(tol), max_epoch(max_epoch), verbose(verbose), iter(0) {
    std::cout << "Maximum iterations: " << ConvergenceMonitor::max_epoch << "\n";
  };

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
inline double logsumexp(Eigen::Array<double, 1, -1>& X, size_t size) {
  double max_value = 0.0;
  max_value = X.maxCoeff();
  if (std::isinf(max_value)) return -INFINITY;
  double acc = 0.0;
  for (size_t i = 0; i < size; i++) {
    acc += std::expl(X(i) - max_value);
  }
  return std::logl(acc) + max_value;
}
inline double logsumexp(double *X, size_t size) {
  double max_value = *std::max_element(X, X + size);
  if (std::isinf(max_value)) return -INFINITY;
  double acc = 0.0;
  for (size_t i = 0; i < size; i++) {
    acc += std::expl(X[i] - max_value);
  }
  return std::logl(acc) + max_value;
}

/*!
 * \brief Normalize probability
 */
inline void log_normalize(Eigen::ArrayXXd& a) {
  Eigen::ArrayXd sum_row(a.rows());
  for (size_t i = 0; i < static_cast<size_t>(a.rows()); i++) {
    Eigen::Array<double, 1, -1> row = a.row(i);
    sum_row(i) = logsumexp(row, static_cast<size_t>(a.cols()));
  }
  a.colwise() -= sum_row;
}

/*!
 * \brief Normailze
 */
inline void normalize(Eigen::ArrayXXd& a) {
  Eigen::ArrayXd row_sum = a.rowwise().sum();
  for (size_t i = 0; i < static_cast<size_t>(row_sum.rows()); i++) {
    row_sum(i) = row_sum(i) ? row_sum(i) : 1;
  }
  a = a.colwise() / row_sum;
}
inline void normalize(Eigen::ArrayXd& a) {
  double _sum = a.sum();
  _sum = _sum ? _sum : 1;
  a /= _sum;
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

void forward(size_t n_observations, size_t n_components,
             const Eigen::ArrayXd& log_stateprob,
             const Eigen::ArrayXXd& log_transmit,
             const Eigen::ArrayXXd& framelogprob,
             Eigen::ArrayXXd& alpha);

void backward(size_t n_observations, size_t n_components,
             const Eigen::ArrayXXd& log_transmit,
             const Eigen::ArrayXXd& framelogprob,
             Eigen::ArrayXXd& beta);

void compute_log_xi_sum(size_t n_observations, size_t n_components,
                        const Eigen::ArrayXXd& alpha,
                        const Eigen::ArrayXXd& log_transmit,
                        const Eigen::ArrayXXd& bwdlattice,
                        const Eigen::ArrayXXd& framelogprob,
                        Eigen::ArrayXXd& log_xi_sum);

void viterbi(size_t n_observations, size_t n_components,
             const Eigen::ArrayXd& log_startprob,
             const Eigen::ArrayXXd& log_transmit,
             const Eigen::ArrayXXd& framelogprob,
             Eigen::ArrayXi& state_sequence,
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
  IterFromIndividualLength(const std::vector<double>& X, const std::vector<size_t>& lengths);
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

inline void log_univariate_normal_density(const std::vector<double>& X,
                                          Eigen::ArrayXd& means,
                                          Eigen::ArrayXd& covars,
                                          Eigen::ArrayXXd& logprob) {
  constexpr double pi = 3.141592653;  
  for (size_t i = 0; i < X.size(); i++) {
    logprob.row(i) = -0.5 * (std::log(2.0 * pi) + covars.log()
                             + Eigen::pow(X[i] - means, 2.0) / covars)
                            .transpose();
  }
}

inline double univariate_normal(const double mean,
                                const double covars, 
                                const int random_seed = -1) {
  static std::random_device seed;
  static std::mt19937 random_number_generator(random_seed == -1 ?
                                              seed() : random_seed);
  std::normal_distribution<double> randn(mean, sqrt(covars));
  
  return randn(random_number_generator);
}
