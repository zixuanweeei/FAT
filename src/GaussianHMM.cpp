#include "../include/GaussianHMM.h"
#include "../include/Kmeans.h"
#include "../include/utils.h"

GaussianHMM::GaussianHMM(int N,
                         ArrayXd means_prior,
                         ArrayXd means_weight,
                         ArrayXd covars_prior,
                         ArrayXd covars_weight,
                         int random_seed, 
                         int max_epoch,
                         double tol, 
                         bool verbose, 
                         double _min_covar)
    : BaseHMM(N, random_seed, max_epoch, tol, verbose),
      means_prior(means_prior),
      means_weight(means_weight),
      covars_prior(covars_prior),
      covars_weight(covars_weight),
      min_covar(ArrayXd::Ones(N) * _min_covar) { 
  double init = 1.0 / N;
  means_ = ArrayXd::Ones(N) * init;
  covars_ = ArrayXd::Ones(N);
}
GaussianHMM::GaussianHMM(int N,
                         double covars_prior,
                         int random_seed, 
                         int max_epoch,
                         double tol, 
                         bool verbose, 
                         double _min_covar)
    : BaseHMM(N, random_seed, max_epoch, tol, verbose),
      means_prior(ArrayXd::Zero(N)),
      means_weight(ArrayXd::Zero(N)),
      covars_prior(ArrayXd::Ones(N) * covars_prior),
      covars_weight(ArrayXd::Ones(N)),
      min_covar(ArrayXd::Ones(N) * _min_covar) { 
  double init = 1.0 / N;
  means_ = ArrayXd::Ones(N) * init;
  covars_ = ArrayXd::Ones(N);
}

void GaussianHMM::_init(const std::vector<double>& X,
                        const std::vector<size_t>& lengths) {
  BaseHMM::_init(X, lengths);
  Kmeans kmeans(BaseHMM::N, 100);
  kmeans.fit(X);
  means_ = *(kmeans.means);
  covars_ = *(kmeans.vars);
}

void GaussianHMM::_compute_log_likelihood(const std::vector<double>& X,
                                          ArrayXXd& logprob) {
  assert(X.size() == static_cast<size_t>(logprob.rows()));
  log_univariate_normal_density(X, means_, covars_, logprob);
}

void GaussianHMM::_generate_sample_from_state(size_t state, 
                                              int random_seed,
                                              std::vector<double>& X) {
  X.push_back(univariate_normal((means_)(state), (covars_)(state), random_seed));
}

void GaussianHMM::_initialize_sufficient_statistics(Stats& stats) {
  BaseHMM::_initialize_sufficient_statistics(stats);
  stats.post = ArrayXd::Zero(N);
  stats.obs = ArrayXd::Zero(N);
  stats.obs_square = ArrayXd::Zero(N);
}

void GaussianHMM::_accumulate_sufficient_statistics(Stats& stats,
                                                    const std::vector<double>& X,
                                                    ArrayXXd& framelogprob,
                                                    ArrayXXd& posteriors,
                                                    ArrayXXd& alpha,
                                                    ArrayXXd& beta) {
  BaseHMM::_accumulate_sufficient_statistics(stats, X,
                                             framelogprob,
                                             posteriors,
                                             alpha,
                                             beta);
  ArrayXd X_obs(X.size());
  for (size_t i = 0; i < X.size(); i++) {
    X_obs(i) = X[i];
  }
  stats.post += posteriors.colwise().sum().transpose();
  stats.obs += (posteriors.matrix().transpose() * X_obs.matrix()).array();
  stats.obs_square += (posteriors.matrix().transpose() * (X_obs * X_obs).matrix()).array();
}

void GaussianHMM::_do_mstep(Stats& stats) {
  BaseHMM::_do_mstep(stats);
  means_ = (means_weight * means_prior + stats.obs) / (means_weight + stats.post);
  ArrayXd meandiff = means_ - means_prior;

  ArrayXd cv_num = means_weight * meandiff * meandiff
                   + stats.obs_square
                   - 2.0 * means_ * stats.obs
                   + means_ * means_ * stats.post;
  ArrayXd cv_den = (covars_weight - 1.0).max(0.0) + stats.post;
  covars_ = (covars_prior + cv_num) / cv_den.max(1e-5);
}

// GaussianHMM::~GaussianHMM() {
//   BaseHMM::~BaseHMM();
// }