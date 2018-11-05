#include "../include/hmm.h"
#include "../include/utils.h"
#include <iostream>
#include <vector>
#include <random>

void BaseHMM::score_samples(const std::vector<double>& X,
                            const std::vector<int>& lengths,
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
    Eigen::ArrayXXd framelogprob(end - start, N);
    _compute_log_likelihood(x, framelogprob);
    _do_forward_pass(framelogprob, &tmp_logprob, Eigen::ArrayXXd(0, 0));
    *logprob += tmp_logprob;
  }
}

void BaseHMM::_decode_viterbi(const std::vector<double>& X,
                              double *logprob,
                              Eigen::ArrayXd& state_sequence) {
  Eigen::ArrayXXd framelogprob(X.size(), N);
  _do_viterbi_pass(framelogprob, logprob, state_sequence);
}

void BaseHMM::_decode_map(const std::vector<double>& X,
                          double *logprob,
                          Eigen::ArrayXd& state_sequence) {
  Eigen::ArrayXXd posteriors(X.size(), N);
  score_samples(X, std::vector<int>(), nullptr, posteriors);
  *logprob = 0;
  for (size_t i = 0; i < X.size(); i++) {
    double tmp_log = posteriors.row(i).maxCoeff(&state_sequence[i]);
    *logprob += tmp_log;
  }
}

void BaseHMM::decode(const std::vector<double>& X, const std::vector<int>& lengths,
                     void (*decoder)(const std::vector<double>&, double*, Eigen::ArrayXd&),
                     double *logprob, Eigen::ArrayXd& state_sequence) {
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
    Eigen::ArrayXd tmp_state_sequence(end - start);
    decoder(x, &tmp_logprob, tmp_state_sequence);
    if (logprob != nullptr) *logprob += tmp_logprob;
    state_sequence.block(start, 0, end, 0) = tmp_state_sequence;
  }
}

void BaseHMM::predict(const std::vector<double>& X, const std::vector<int>& lengths,
                      Eigen::ArrayXd& state_sequence) {
  decode(X, lengths, this->_decode_viterbi, nullptr, state_sequence);
}

void BaseHMM::predict_proba(const std::vector<double>& X,
                            const std::vector<int>& lengths,
                            Eigen::ArrayXXd& posteriors) {
  score_samples(X, lengths, nullptr, posteriors);
}

void BaseHMM::sample(int n_samples, int random_seed, std::vector<double>& X,
                     Eigen::ArrayXd& state_sequence) {
  Eigen::ArrayXd pi_cdf(pi->rows());
  pi_cdf(0) = (*pi)(0);
  for (size_t i = 1; i < pi->rows(); i++) {
    pi_cdf(i) = pi_cdf(i - 1) + (*pi)(i);
  }

  Eigen::ArrayXXd A_cdf(A->rows(), A->cols());
  A_cdf.row(0) = A->row(0);
  for (size_t i = 1; i < A->rows(); i++) {
    A_cdf.row(i) = A_cdf.row(i - 1) + A->row(i);
  }
  std::default_random_engine random_generator(random_seed);
  std::uniform_real_distribution proba(0., 1.);

  size_t currstate = 0;
  size_t currpos = 0;
  double state_proba = proba(random_generator);
  Eigen::ArrayXd state_compare = (pi_cdf > state_proba);
  state_compare.maxCoeff(&currstate);
  state_sequence(currpos) = currstate;
  _generate_sample_from_state(currstate, random_seed, X);
  while (++currpos < n_samples) {
    state_proba = proba(random_generator);
    state_compare = pi_cdf > state_proba;
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
                               Eigen::ArrayXd& state_sequence) {
  size_t n_samples = framelogprob.rows();
  size_t n_components = framelogprob.cols();

  const Eigen::ArrayXd log_pi = pi->log();
  const Eigen::ArrayXXd log_A = A->log();
  viterbi(n_samples, n_components, log_pi, log_A, framelogprob,
          state_sequence, logprob);
}

void BaseHMM::_do_forward_pass(Eigen::ArrayXXd& framelogprob, double *logprob,
                               Eigen::ArrayXXd& alpha) {
  size_t n_samples = framelogprob.rows();
  size_t n_components = framelogprob.cols();
  const Eigen::ArrayXd log_pi = pi->log();
  const Eigen::ArrayXXd log_A = A->log();
  forward(n_samples, n_components, log_pi, log_A, framelogprob, alpha);
  Eigen::Array<double, 1, -1> row = alpha.row(alpha.rows() - 1);
  *logprob = logsumexp(row, alpha.cols());
}

void BaseHMM::_do_backward_pass(Eigen::ArrayXXd& framelogprob, 
                                Eigen::ArrayXXd& beta) {
  size_t n_samples = framelogprob.rows();
  size_t n_components = framelogprob.cols();
  const Eigen::ArrayXd log_pi = pi->log();
  const Eigen::ArrayXXd log_A = A->log();
  forward(n_samples, n_components, log_pi, log_A, framelogprob, beta);
}

void BaseHMM::_compute_posteriors(Eigen::ArrayXXd& alpha, Eigen::ArrayXXd& beta, 
                                  Eigen::ArrayXXd& log_gamma) {
  Eigen::ArrayXXd tmp_log_gamma = alpha + beta;
  log_normalize(tmp_log_gamma);
  log_gamma = tmp_log_gamma.exp();
}

void BaseHMM::_init(const std::vector<double>& X) {
  double init_value = 1./N;
  pi = new Eigen::ArrayXd(N);
  *pi = init_value * Eigen::ArrayXd::Ones(N);
  A = new Eigen::ArrayXXd(N, N);
  *A = init_value * Eigen::ArrayXXd::Ones(N, N);
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
  size_t n_samples = framelogprob.rows();
  size_t n_components = framelogprob.cols();

  Eigen::ArrayXXd log_xi_sum(n_components, n_components);
  log_xi_sum = -INFINITY * Eigen::ArrayXXd::Ones(n_components, n_components);
  const Eigen::ArrayXXd log_trans = trans.log();
  compute_log_xi_sum(n_samples, n_components, alpha, log_trans, beta, 
                     framelogprob, log_xi_sum);
  trans += log_xi_sum.exp();
}

void BaseHMM::_do_mstep(size_t n_observations, Eigen::ArrayXd& start,
                        Eigen::ArrayXXd& trans) {
  Eigen::ArrayXd pi_ = (*pi) + start - Eigen::ArrayXd::Ones(pi_.rows());
  for (size_t i = 0; i < pi_.rows(); i++) {
    (*pi)(i) = (*pi)(i) == 0 ? (*pi)(i) : pi_(i);
  }
  normalize(*pi);
  
  Eigen::ArrayXXd A_ = *A + trans - 1.0;
  for (size_t i = 0; i < A_.rows(); i++) {
    for (size_t j = 0; j < A_.cols(); j++) {
      (*A)(i, j) = (*A)(i, j) == 0 ? (*A)(i, j) : A_(i, j);
    }
  }
  normalize(*A);
}

BaseHMM::~BaseHMM() {
  delete A;
  delete pi;
}
