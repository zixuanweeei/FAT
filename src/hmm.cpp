#include "../include/hmm.h"
#include "../include/utils.h"
#include <iostream>
#include <vector>
#include <random>

void BaseHMM::score_samples(const std::vector<double>& X,
                            const std::vector<int>& lengths,
                            double *logprob, Eigen::MatrixXd& posteriors) {
  size_t n_observations = X.size();
  size_t n_individuals = lengths.size();
  if (n_individuals == 0) n_individuals = 1;
  if (logprob != nullptr) *logprob = 0;
  IterFromIndividualLength iter(X, lengths);
  for (size_t i = 0; i < n_individuals; i++) {
    size_t start = iter.get_start();
    size_t end = iter.get_end();
    std::vector<double> x(X.begin() + start, X.begin() + end);

    Eigen::MatrixXd framelogprob(end - start, N);
    _compute_log_likelihood(x, framelogprob);

    Eigen::MatrixXd alpha(end - start, N);
    double tmp_logprob;
    _do_forward_pass(framelogprob, &tmp_logprob, alpha);
    if (logprob != nullptr) *logprob += tmp_logprob;

    Eigen::MatrixXd beta(end - start, N);
    _do_backward_pass(framelogprob, beta);
    Eigen::MatrixXd tmp_posteriors(end - start, N);
    _compute_posteriors(alpha, beta, tmp_posteriors);
    for (size_t j = 0; j < end - start; j++) {
      posteriors.row(start + j) = tmp_posteriors.row(j);
    }
  }
}

void BaseHMM::score(const std::vector<double>& X,
                    const std::vector<int>& lengths,
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
    Eigen::MatrixXd framelogprob(end - start, N);
    _compute_log_likelihood(x, framelogprob);
    _do_forward_pass(framelogprob, &tmp_logprob, Eigen::MatrixXd(0, 0));
    *logprob += tmp_logprob;
  }
}

void BaseHMM::_decode_viterbi(const std::vector<double>& X,
                              double *logprob,
                              Eigen::VectorXd& state_sequence) {
  Eigen::MatrixXd framelogprob(X.size(), N);
  _do_viterbi_pass(framelogprob, logprob, state_sequence);
}

void BaseHMM::_decode_map(const std::vector<double>& X,
                          double *logprob,
                          Eigen::VectorXd& state_sequence) {
  Eigen::MatrixXd posteriors(X.size(), N);
  score_samples(X, std::vector<int>(), nullptr, posteriors);
  *logprob = 0;
  for (size_t i = 0; i < X.size(); i++) {
    double tmp_log = posteriors.row(i).maxCoeff(&state_sequence[i]);
    *logprob += tmp_log;
  }
}

void BaseHMM::decode(const std::vector<double>& X, const std::vector<int>& lengths,
                     void (*decoder)(const std::vector<double>&, double*, Eigen::VectorXd&),
                     double *logprob, Eigen::VectorXd& state_sequence) {
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
    Eigen::VectorXd tmp_state_sequence(end - start);
    decoder(x, &tmp_logprob, tmp_state_sequence);
    if (logprob != nullptr) *logprob += tmp_logprob;
    state_sequence.block(start, 0, end, 0) = tmp_state_sequence;
  }
}

void BaseHMM::predict(const std::vector<double>& X, const std::vector<int>& lengths,
                      Eigen::VectorXd& state_sequence) {
  decode(X, lengths, this->_decode_viterbi, nullptr, state_sequence);
}

void BaseHMM::predict_proba(const std::vector<double>& X,
                            const std::vector<int>& lengths,
                            Eigen::MatrixXd& posteriors) {
  score_samples(X, lengths, nullptr, posteriors);
}

void BaseHMM::sample(int n_samples, int random_seed, std::vector<double>& X,
                     Eigen::VectorXd& state_sequence) {
  Eigen::VectorXd pi_cdf(pi->rows());
  pi_cdf(0) = (*pi)(0);
  for (size_t i = 1; i < pi->rows(); i++) {
    pi_cdf(i) = pi_cdf(i - 1) + (*pi)(i);
  }

  Eigen::MatrixXd A_cdf(A->rows(), A->cols());
  A_cdf.row(0) = A->row(0);
  for (size_t i = 1; i < A->rows(); i++) {
    A_cdf.row(i) = A_cdf.row(i - 1) + A->row(i);
  }
  std::default_random_engine random_generator(random_seed);
  std::uniform_real_distribution proba(0., 1.);

  size_t currstate = 0;
  size_t currpos = 0;
  double state_proba = proba(random_generator);
  Eigen::VectorXd state_compare = (pi_cdf.array() > state_proba);
  state_compare.maxCoeff(&currstate);
  state_sequence(currpos) = currstate;
  _generate_sample_from_state(currstate, random_seed, X);
  while (++currpos < n_samples) {
    state_proba = proba(random_generator);
    state_compare = pi_cdf.array() > state_proba;
    state_compare.maxCoeff(&currstate);
    state_sequence(currpos) = currstate;
    _generate_sample_from_state(currstate, random_seed, X);
  }
}

void BaseHMM::fit(const std::vector<double>& X, const std::vector<int>& lengths) {
  _init(X);
  ConvergenceMonitor moniter(1e-4, 10, true);
  IterFromIndividualLength iter(X, lengths);
  size_t n_observations;
  size_t n_individuals = lengths.size() ? lengths.size() : 1;
  Eigen::VectorXd *tmp_pi;
  Eigen::MatrixXd *tmp_A;
  for (size_t i = 0; i < max_epoch; i++) {
    _initialize_sufficient_statistics(&n_observations, tmp_pi, tmp_A);
    double curr_logprob = 0;
    for (size_t j = 0; j < n_individuals; j++) {
      size_t start = iter.get_start();
      size_t end = iter.get_end();
      std::vector<double> x(X.begin() + start, X.begin() + end);

      Eigen::MatrixXd framelogprob(end - start, N);
      Eigen::MatrixXd alpha(end - start, N);
      _compute_log_likelihood(x, framelogprob);
      double tmplogprob = 0;
      _do_forward_pass(framelogprob, &tmplogprob, alpha);
      curr_logprob += tmplogprob;

      Eigen::MatrixXd beta(end - start, N);
      _do_backward_pass(framelogprob, beta);

      Eigen::MatrixXd posteriors(end - start, N);
      _compute_posteriors(alpha, beta, posteriors);

      _accumulate_sufficient_statistics(n_observations, *tmp_pi, *tmp_A, x, 
                                        framelogprob, posteriors, alpha, beta);
    }
    _do_mstep(n_observations, *tmp_pi, *tmp_A);
    delete tmp_pi;
    delete tmp_A;

    if (moniter.report(curr_logprob)) break;
  }
}

void BaseHMM::_do_viterbi_pass(Eigen::MatrixXd& framelogprob, double *logprob,
                               Eigen::VectorXd& state_sequence) {
  size_t n_samples = framelogprob.rows();
  size_t n_components = framelogprob.cols();

  Eigen::MatrixXd log_pi = pi->log();
  Eigen::MatrixXd log_A = A->log();
  viterbi(n_samples, n_components, log_pi, log_A, framelogprob,
          state_sequence, logprob);
}