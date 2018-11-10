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

void GaussianHMM::_initialize_sufficient_statistics(size_t *n_observations,
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
  ArrayXd X_obs(X.size());
  for (size_t i = 0; i < X.size(); i++) {
    X_obs(i) = X[i];
  }
  post += posteriors.colwise().sum().transpose();
  obs += (posteriors.matrix().transpose() * X_obs.matrix()).array();
  obs_square += (posteriors.matrix().transpose() * (X_obs * X_obs).matrix()).array();
}

void GaussianHMM::fit(const std::vector<double>& X, const std::vector<size_t>& lengths) {
  _init(X, lengths);
  ConvergenceMonitor moniter(1e-4, max_epoch, true);
  IterFromIndividualLength iter(X, lengths);
  size_t n_observations;
  size_t n_individuals = lengths.size() ? lengths.size() : 1;
  Eigen::ArrayXd tmp_pi;
  Eigen::ArrayXXd tmp_A;
  Eigen::ArrayXd post;
  Eigen::ArrayXd obs;
  Eigen::ArrayXd obs_square;
  for (size_t i = 0; i < max_epoch; i++) {
    _initialize_sufficient_statistics(&n_observations, tmp_pi, tmp_A, post, obs, obs_square);
    double curr_logprob = 0;
    for (size_t j = 0; j < n_individuals; j++) {
      size_t start = iter.get_start();
      size_t end = iter.get_end();
      std::vector<double> x(X.begin() + start, X.begin() + end);

      Eigen::ArrayXXd framelogprob(end - start, N);
      Eigen::ArrayXXd alpha(end - start, N);
      _compute_log_likelihood(x, framelogprob);
      double tmplogprob = 0;
      _do_forward_pass(framelogprob, &tmplogprob, alpha);
      curr_logprob += tmplogprob;

      Eigen::ArrayXXd beta(end - start, N);
      _do_backward_pass(framelogprob, beta);

      Eigen::ArrayXXd posteriors(end - start, N);
      _compute_posteriors(alpha, beta, posteriors);

      _accumulate_sufficient_statistics(&n_observations, tmp_pi, tmp_A, post, obs, obs_square, x, 
                                        framelogprob, posteriors, alpha, beta);
    }
    _do_mstep(n_observations, tmp_pi, tmp_A, post, obs, obs_square);

    if (moniter.report(curr_logprob)) break;
  }
}

void GaussianHMM::_do_mstep(size_t n_observations,
                            ArrayXd &start,
                            ArrayXXd &trans,
                            ArrayXd &post,
                            ArrayXd &obs,
                            ArrayXd &obs_square) {
  BaseHMM::_do_mstep(n_observations, start, trans);
  means_ = (means_weight * means_prior + obs) / (means_weight + post);

  ArrayXd covars_prior_ = covars_prior;
  ArrayXd covars_weight_ = covars_weight;
  ArrayXd meandiff = means_ - means_prior;

  ArrayXd cv_num = means_weight * pow(meandiff, 2.0)
                   + obs_square
                   - 2.0 * (means_) * obs
                   + pow(means_, 2.0) * post;
  ArrayXd cv_den = (covars_weight_ - 1.0).max(0.0) + post;
  covars_ = (covars_prior + cv_num) / cv_den.max(1e-5);
}

// GaussianHMM::~GaussianHMM() {
//   BaseHMM::~BaseHMM();
// }