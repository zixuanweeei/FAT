#include "../include/GaussianHMM.h"
#include "../include/Kmeans.h"
#include "../include/utils.h"

GaussianHMM::GaussianHMM(int N,
                         ArrayXd means_prior,
                         ArrayXd means_weight,
                         ArrayXd covars_prior,
                         ArrayXd covars_weight,
                         int random_seed = -1, 
                         int max_epoch = 10,
                         double tol = 1e-4, 
                         bool verbose = true, 
                         double _min_covar = 1e-3)
    : BaseHMM(N, random_seed, max_epoch, tol, verbose),
      means_prior(means_prior),
      means_weight(means_weight),
      covars_prior(covars_prior),
      covars_weight(covars_weight),
      min_covar(ArrayXd::Ones(N) * _min_covar) { }
GaussianHMM::GaussianHMM(int N,
                         ArrayXd means_prior,
                         double covars_prior,
                         int random_seed = -1, 
                         int max_epoch = 10,
                         double tol = 1e-4, 
                         bool verbose = true, 
                         double _min_covar = 1e-3)
    : BaseHMM(N, random_seed, max_epoch, tol, verbose),
      means_prior(means_prior),
      means_weight(ArrayXd::Ones(N) / N),
      covars_prior(ArrayXd::Ones(N) * covars_prior),
      covars_weight(ArrayXd::Ones(N) / N),
      min_covar(ArrayXd::Ones(N) * _min_covar) { }

void GaussianHMM::_init(const std::vector<double>& X,
                        const std::vector<double>& lengths) {
  
  Kmeans kmeans(BaseHMM::N, 100);
  kmeans.fit(X);
  means_ = *(kmeans.means);
  covars_ = *(kmeans.vars);
}

void GaussianHMM::_compute_log_likelihood(const std::vector<double>& X,
                                          ArrayXXd& logprob) {
  log_univariate_normal_density(X, means_, covars_, logprob);
}

void GaussianHMM::_generate_sample_from_state(size_t state, 
                                              int random_seed,
                                              std::vector<double>& X) {
  X.push_back(univariate_normal(means_(state), covars_(state), random_seed));
}

void GaussianHMM::_init_sufficient_statistics(size_t *n_observations,
                                              ArrayXd &start,
                                              ArrayXXd &trans,
                                              ArrayXd &post,
                                              ArrayXd &obs,
                                              ArrayXd &obs_square) {
  BaseHMM::_initialize_sufficient_statistics(n_observations,
                                             start,
                                             trans);
  post = ArrayXd::Zero(N);
  obs = ArrayXd::Zero(N);
  obs_square = ArrayXd::Zero(N);
}

void GaussianHMM::_accumulate_sufficient_statistics(size_t *n_observations,
                                                    ArrayXd &start,
                                                    ArrayXXd &trans,
                                                    ArrayXd &post,
                                                    ArrayXd &obs,
                                                    ArrayXd &obs_square,
                                                    const std::vector<double>& X,
                                                    ArrayXXd& framelogprob,
                                                    ArrayXXd& posteriors,
                                                    ArrayXXd& alpha,
                                                    ArrayXXd& beta) {
  BaseHMM::_accumulate_sufficient_statistics(n_observations,
                                             start, trans, X,
                                             framelogprob,
                                             posteriors,
                                             alpha,
                                             beta);
  post += posteriors.colwise().sum();
  obs += (posteriors.matrix().transpose() * obs.matrix()).array();
  obs_square += (posteriors.matrix().transpose() * obs_square.matrix()).array();
}

void GaussianHMM::_do_mstep(size_t n_observations,
                            ArrayXd &start,
                            ArrayXXd &trans,
                            ArrayXd &post,
                            ArrayXd &obs,
                            ArrayXd &obs_square) {
  BaseHMM::_do_mstep(n_observations, start, trans);
  ArrayXd means_prior_ = means_prior;
  ArrayXd means_weight_ = means_weight;
  means_ = (means_weight_ * means_prior_ + obs) / (means_weight_ + post);

  ArrayXd covars_prior_ = covars_prior;
  ArrayXd covars_weight_ = covars_weight;
  ArrayXd meandiff = means_ - means_prior_;

  ArrayXd cv_num = means_weight_ * pow(meandiff, 2.0) + obs_square
                   - 2*means_ * obs
                   + pow(means_, 2.0) * post;
  ArrayXd cv_den = (covars_weight_ - 1).max(0.0) + post;
  covars_ = (covars_prior + cv_num) / cv_den.max(1e-5);
}