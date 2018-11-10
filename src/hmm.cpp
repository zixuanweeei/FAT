#include "../include/hmm.h"
#include "../include/utils.h"
#include <iostream>
#include <vector>
#include <random>
typedef void (BaseHMM::*decoder)(const std::vector<double>&, double *, Eigen::ArrayXi&);

void BaseHMM::score_samples(const std::vector<double>& X,
                            const std::vector<size_t>& lengths,
                            double *logprob, Eigen::ArrayXXd& posteriors) {
  size_t n_observations = X.size();
  size_t n_individuals = lengths.size();
  if (n_individuals == 0) n_individuals = 1;
  if (logprob != nullptr) *logprob = 0;
  IterFromIndividualLength iter(X, lengths);
  for (size_t i = 0; i < n_individuals; i++) {
    size_t start = iter.get_start();
    size_t end = iter.get_end();
    std::vector<double> x(X.begin() + start, X.begin() + end);

    Eigen::ArrayXXd framelogprob(end - start, N);
    _compute_log_likelihood(x, framelogprob);

    Eigen::ArrayXXd alpha(end - start, N);
    double tmp_logprob;
    _do_forward_pass(framelogprob, &tmp_logprob, alpha);
    if (logprob != nullptr) *logprob += tmp_logprob;

    Eigen::ArrayXXd beta(end - start, N);
    _do_backward_pass(framelogprob, beta);
    Eigen::ArrayXXd tmp_posteriors(end - start, N);
    _compute_posteriors(alpha, beta, tmp_posteriors);
    for (size_t j = 0; j < end - start; j++) {
      posteriors.row(start + j) = tmp_posteriors.row(j);
    }
  }
}

void BaseHMM::score(const std::vector<double>& X,
                    const std::vector<size_t>& lengths,
                    double *logprob) {
  IterFromIndividualLength iter(X, lengths);
  size_t n_individuals = lengths.size();
  if (n_individuals == 0) n_individuals = 1;
  *logprob = 0;
  for (size_t i = 0; i < n_individuals; i++) {
    size_t start = iter.get_start();
    size_t end = iter.get_end();
    std::vector<double> x(X.begin() + start, X.begin() + end);

    double tmp_logprob;
    Eigen::ArrayXXd framelogprob(end - start, N);
    Eigen::ArrayXXd alpha(end - start, N);
    _compute_log_likelihood(x, framelogprob);
    _do_forward_pass(framelogprob, &tmp_logprob, alpha);
    *logprob += tmp_logprob;
  }
}

void BaseHMM::_decode_viterbi(const std::vector<double>& X,
                              double *logprob,
                              Eigen::ArrayXi& state_sequence) {
  Eigen::ArrayXXd framelogprob(X.size(), N);
  _do_viterbi_pass(framelogprob, logprob, state_sequence);
}

void BaseHMM::_decode_map(const std::vector<double>& X,
                          double *logprob,
                          Eigen::ArrayXi& state_sequence) {
  Eigen::ArrayXXd posteriors(X.size(), N);
  score_samples(X, std::vector<size_t>(), nullptr, posteriors);
  *logprob = 0;
  for (size_t i = 0; i < X.size(); i++) {
    double tmp_log = posteriors.row(i).maxCoeff(&state_sequence[i]);
    *logprob += tmp_log;
  }
}

void BaseHMM::decode(const std::vector<double>& X, const std::vector<size_t>& lengths,
                     decoder d,
                     double *logprob, Eigen::ArrayXi& state_sequence) {
  size_t n_observations = X.size();
  size_t n_individuals = lengths.size();
  if (n_individuals == 0) n_individuals = 1;
  *logprob = 0;
  IterFromIndividualLength iter(X, lengths);
  for (size_t i = 0; i < n_individuals; i++) {
    size_t start = iter.get_start();
    size_t end = iter.get_end();
    std::vector<double> x(X.begin() + start, X.begin() + end);
    double tmp_logprob = 0;
    Eigen::ArrayXi tmp_state_sequence(static_cast<int>(end - start));
    (this->*d)(x, &tmp_logprob, tmp_state_sequence);
    if (logprob != nullptr) *logprob += tmp_logprob;
    state_sequence.block(start, 0, end - start, 1) = tmp_state_sequence;
  }
}

void BaseHMM::predict(const std::vector<double>& X, const std::vector<size_t>& lengths,
                      Eigen::ArrayXi& state_sequence) {
  decode(X, lengths, &BaseHMM::_decode_viterbi, nullptr, state_sequence);
}

void BaseHMM::predict_proba(const std::vector<double>& X,
                            const std::vector<size_t>& lengths,
                            Eigen::ArrayXXd& posteriors) {
  score_samples(X, lengths, nullptr, posteriors);
}

void BaseHMM::sample(size_t n_samples, int random_seed, std::vector<double>& X,
                     Eigen::ArrayXi& state_sequence) {
  Eigen::ArrayXd pi_cdf(pi.rows());
  pi_cdf(0) = pi(0);
  for (size_t i = 1; i < static_cast<size_t>(pi.rows()); i++) {
    pi_cdf(i) = pi_cdf(i - 1) + pi(i);
  }

  Eigen::ArrayXXd A_cdf(A.rows(), A.cols());
  A_cdf.col(0) = A.col(0);
  for (size_t i = 1; i < static_cast<size_t>(A.cols()); i++) {
    A_cdf.col(i) = A_cdf.col(i - 1) + A.col(i);
  }
  std::random_device seed;
  std::default_random_engine random_generator(random_seed == -1 ? seed() : random_seed);
  std::uniform_real_distribution<double> proba(0., 1.);

  int currstate = 0;
  int currpos = 0;
  double state_proba = proba(random_generator);
  Eigen::ArrayXd state_compare = (pi_cdf > state_proba).cast<double>();
  state_compare.maxCoeff(&currstate);
  state_sequence(currpos) = currstate;
  _generate_sample_from_state(currstate, random_seed, X);
  while (++currpos < static_cast<int>(n_samples)) {
    state_proba = proba(random_generator);
    state_compare = (A_cdf.row(currstate) > state_proba).cast<double>();
    state_compare.maxCoeff(&currstate);
    state_sequence(currpos) = currstate;
    _generate_sample_from_state(currstate, random_seed, X);
  }
}

void BaseHMM::fit(const std::vector<double>& X, const std::vector<size_t>& lengths) {
  _init(X, lengths);
  ConvergenceMonitor moniter(1e-4, max_epoch, true);
  IterFromIndividualLength iter(X, lengths);
  size_t n_observations;
  size_t n_individuals = lengths.size() ? lengths.size() : 1;
  Eigen::ArrayXd tmp_pi;
  Eigen::ArrayXXd tmp_A;
  for (size_t i = 0; i < max_epoch; i++) {
    _initialize_sufficient_statistics(&n_observations, tmp_pi, tmp_A);
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

      _accumulate_sufficient_statistics(&n_observations, tmp_pi, tmp_A, x, 
                                        framelogprob, posteriors, alpha, beta);
    }
    _do_mstep(n_observations, tmp_pi, tmp_A);

    if (moniter.report(curr_logprob)) break;
  }
}

void BaseHMM::_do_viterbi_pass(Eigen::ArrayXXd& framelogprob, double *logprob,
                               Eigen::ArrayXi& state_sequence) {
  size_t n_samples = static_cast<size_t>(framelogprob.rows());
  size_t n_components = static_cast<size_t>(framelogprob.cols());

  const Eigen::ArrayXd log_pi = pi.log();
  const Eigen::ArrayXXd log_A = A.log();
  viterbi(n_samples, n_components, log_pi, log_A, framelogprob,
          state_sequence, logprob);
}

void BaseHMM::_do_forward_pass(Eigen::ArrayXXd& framelogprob, double *logprob,
                               Eigen::ArrayXXd& alpha) {
  size_t n_samples = static_cast<size_t>(framelogprob.rows());
  size_t n_components = static_cast<size_t>(framelogprob.cols());
  const Eigen::ArrayXd log_pi = pi.log();
  const Eigen::ArrayXXd log_A = A.log();
  forward(n_samples, n_components, log_pi, log_A, framelogprob, alpha);
  Eigen::Array<double, 1, -1> row = alpha.row(alpha.rows() - 1);
  *logprob = logsumexp(row, alpha.cols());
}

void BaseHMM::_do_backward_pass(Eigen::ArrayXXd& framelogprob, 
                                Eigen::ArrayXXd& beta) {
  size_t n_samples = static_cast<size_t>(framelogprob.rows());
  size_t n_components = static_cast<size_t>(framelogprob.cols());
  const Eigen::ArrayXXd log_A = A.log();
  backward(n_samples, n_components, log_A, framelogprob, beta);
}

void BaseHMM::_compute_posteriors(Eigen::ArrayXXd& alpha, Eigen::ArrayXXd& beta, 
                                  Eigen::ArrayXXd& gamma) {
  Eigen::ArrayXXd tmp_log_gamma = alpha + beta;
  log_normalize(tmp_log_gamma);
  gamma = tmp_log_gamma.exp();
}

void BaseHMM::_init(const std::vector<double>& X, 
                    const std::vector<size_t>& lengths) { 
  double init = 1.0 / N;
  pi = Eigen::ArrayXd::Ones(N) * init;
  A = Eigen::ArrayXXd::Ones(N, N) * init;
}

void BaseHMM::_initialize_sufficient_statistics(size_t *n_observations,
                                                Eigen::ArrayXd &start,
                                                Eigen::ArrayXXd &trans) {
  *n_observations = 0;
  start = Eigen::ArrayXd::Zero(N);
  trans = Eigen::ArrayXXd::Zero(N, N);
}

void BaseHMM::_accumulate_sufficient_statistics(size_t *n_observations,
                                                Eigen::ArrayXd& start,
                                                Eigen::ArrayXXd& trans,
                                                const std::vector<double>& X,
                                                Eigen::ArrayXXd& framelogprob,
                                                Eigen::ArrayXXd& posteriors,
                                                Eigen::ArrayXXd& alpha,
                                                Eigen::ArrayXXd& beta) {
  (*n_observations) ++;
  start += posteriors.row(0).transpose();
  size_t n_samples = static_cast<size_t>(framelogprob.rows());
  size_t n_components = static_cast<size_t>(framelogprob.cols());

  Eigen::ArrayXXd log_xi_sum(n_components, n_components);
  log_xi_sum = -INFINITY * Eigen::ArrayXXd::Ones(n_components, n_components);
  const Eigen::ArrayXXd log_trans = A.log();
  compute_log_xi_sum(n_samples, n_components, alpha, log_trans, beta, 
                     framelogprob, log_xi_sum);
  trans += log_xi_sum.exp();
}

void BaseHMM::_do_mstep(size_t n_observations, Eigen::ArrayXd& start,
                        Eigen::ArrayXXd& trans) {
  for (size_t i = 0; i < static_cast<size_t>(start.rows()); i++) {
    pi(i) = pi(i) <= 1e-16 ? pi(i) : start(i);
  }
  normalize(pi);

  for (size_t i = 0; i < static_cast<size_t>(trans.rows()); i++) {
    for (size_t j = 0; j < static_cast<size_t>(trans.cols()); j++) {
      A(i, j) = A(i, j) <= 1e-16 ? A(i, j) : trans(i, j);
    }
  }
  normalize(A);
}

// BaseHMM::~BaseHMM() {
  
// }
