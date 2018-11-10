#pragma once
#include "hmm.h"
using namespace Eigen;

struct GaussianHMM : public BaseHMM {
  ArrayXd means_;
  ArrayXd covars_;
  ArrayXd min_covar;
  ArrayXd means_prior;
  ArrayXd means_weight;
  ArrayXd covars_prior;
  ArrayXd covars_weight;
  size_t n_features = 1;

  GaussianHMM(int N,
              ArrayXd means_prior,
              ArrayXd means_weight,
              ArrayXd covars_prior,
              ArrayXd covars_weight, 
              int random_seed = 47, 
              int max_epoch = 100,
              double tol = 1e-4, 
              bool verbose = true, 
              double min_covar = 1e-3);
  GaussianHMM(int N,
              double covars_prior = 1e-2,
              int random_seed = 47, 
              int max_epoch = 100,
              double tol = 1e-4, 
              bool verbose = true, 
              double min_covar = 1e-3);

  /*!
  * \brief Initialize
  */
  void _init(const std::vector<double>& X,
             const std::vector<size_t>& lengths);

  /*!
   * \brief Compute log likelihood of Gaussian Distribution
   */
  void _compute_log_likelihood(const std::vector<double>& X,
                               ArrayXXd& logprob);


  /*!
   * \brief Generate sample from state using Gaussian Distribution
   */
  void _generate_sample_from_state(size_t state,
                                   int random_state,
                                   std::vector<double>& X);
  
  /*!
   * \brief Initialize sufficient statistics
   */
  void _initialize_sufficient_statistics(Stats& stats);
  
  void _accumulate_sufficient_statistics(Stats& stats,
                                         const std::vector<double>& X,
                                         ArrayXXd& framelogprob,
                                         ArrayXXd& posteriors,
                                         ArrayXXd& alpha,
                                         ArrayXXd& beta);

  void _do_mstep(Stats& stats);

  // ~GaussianHMM();
};
