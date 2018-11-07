#include "../include/utils.h"
#include <iostream>

bool ConvergenceMonitor::report(double logprob) {
  if (verbose) {
    double delta = history.size() ? (logprob - history.back()) : std::nan("1");
    printf("Epoch[%04zd] - logrob:%6.4f, delta:%6.4f\n", iter, logprob, delta);
  }

  history.push_back(logprob);
  iter++;

  return (iter == max_epoch || 
          (iter > 1 && std::abs(history[iter - 1] - history[iter - 2]) < tol));
}

void forward(size_t n_observations, size_t n_components,
             const Eigen::ArrayXd& log_startprob,
             const Eigen::ArrayXXd& log_transmit,
             const Eigen::ArrayXXd& framelogprob,
             Eigen::ArrayXXd& alpha) {
  size_t t, i, j;
  double *work_buffer = new double[n_components]{0.0};

  for (i = 0; i < n_components; i++) {
    alpha(0, i) = log_startprob[i] + framelogprob(0, i);
  }
  for (t = 1; t < n_observations; t++) {
    for (j = 0; j < n_components; j++) {
      for (i = 0; i < n_components; i++) {
        work_buffer[i] = alpha(t - 1, i) + log_transmit(i, j);
      }
      alpha(t, j) = logsumexp(work_buffer, n_components) + framelogprob(t, j);
    }
  }
  delete [] work_buffer;
}

void backward(size_t n_observations, size_t n_components,
              const Eigen::ArrayXXd& log_transmit,
              const Eigen::ArrayXXd& framelogprob,
              Eigen::ArrayXXd& beta) {
  size_t i, j;
  int t;
  double *work_buffer = new double[n_components]{0.0};

  for (i = 0; i < n_components; i++) {
    beta(n_observations - 1, i) = 0.0;
  }
  for (t = static_cast<int>(n_observations - 2); t >= 0; t--) {
    for (i = 0; i < n_components; i++) {
      for (j = 0; j < n_components; j++) {
        work_buffer[j] = log_transmit(i, j) + framelogprob(t + 1, j)
                         + beta(t + 1, j);
      }
      beta(t, i) = logsumexp(work_buffer, n_components);
    }
  }
  delete [] work_buffer;
}

void compute_log_xi_sum(size_t n_observations, size_t n_components,
                        const Eigen::ArrayXXd& alpha,
                        const Eigen::ArrayXXd& log_transmit,
                        const Eigen::ArrayXXd& beta,
                        const Eigen::ArrayXXd& framelogprob,
                        Eigen::ArrayXXd& log_xi_sum) {
  size_t t, i, j;
  double **work_buffer = new double*[n_components];
  for (size_t i = 0; i < n_components; i++) {
    work_buffer[i] = new double[n_components]{0.0};
  }
  Eigen::Array<double, 1, -1> row = alpha.row(n_observations - 1);
  double logprob = logsumexp(row, n_components);

  for (t = 0; t < n_observations - 1; t++) {
    for (i = 0; i < n_components; i++) {
      for (j = 0; j < n_components; j++) {
        work_buffer[i][j] = alpha(t, i)
                            + log_transmit(i, j)
                            + framelogprob(t, j)
                            + beta(t, j)
                            - logprob;
      }
    }
    for (size_t ii = 0; ii < n_components; ii++) {
      for (size_t jj = 0; jj < n_components; jj++) {
        log_xi_sum(ii, jj) = logaddexp(log_xi_sum(ii, jj), work_buffer[ii][jj]);
      }
    }
  }

  for (i = 0; i < n_components; i++) {
    delete [] work_buffer[i];
  }
  delete [] work_buffer;
}

void viterbi(size_t n_observations, size_t n_components,
             const Eigen::ArrayXd& log_startprob,
             const Eigen::ArrayXXd& log_transmit,
             const Eigen::ArrayXXd& framelogprob,
             Eigen::ArrayXi& state_sequence,
             double *logprob) {
  size_t i, j, t, where_from;

  double **viterbi_lattice = new double*[n_observations];
  for (size_t i = 0; i < n_observations; i++) {
    viterbi_lattice[i] = new double[n_components]{0.0};
  }
  double *work_buffer = new double[n_components]{0.0};

  for (i = 0; i < n_components; i++) {
    viterbi_lattice[0][i] = log_startprob[i] + framelogprob(0, i);
  }
  for (t = 1; t < n_observations - 1; t++) {
    for (i = 0; i < n_components; i++) {
      for (j = 0; j < n_components; j++) {
        work_buffer[j] = log_transmit(j, i)
                         + viterbi_lattice[t - 1][j];
      }
      viterbi_lattice[t][i] = *std::max_element(work_buffer,
                                                work_buffer + n_components)
                              + framelogprob(t, i);
    }
  }

  /* Observation traceback */
  where_from = std::max_element(viterbi_lattice[n_observations - 1], 
                                viterbi_lattice[n_observations - 1] + n_components)
               - viterbi_lattice[n_observations - 1];
  state_sequence[n_observations - 1] = static_cast<int>(where_from);
  *logprob = viterbi_lattice[n_observations - 1][where_from];

  for (t = n_observations - 2; t >= 0; t--) {
    for (i = 0; i < n_components; i++) {
      work_buffer[i] = viterbi_lattice[t][i] + log_transmit(i, where_from);
    }
    where_from = std::max_element(work_buffer, work_buffer + n_components)
                 - work_buffer;
    state_sequence[t] = static_cast<int>(where_from);
  }
  free_mat(viterbi_lattice, n_observations, n_components);
  delete [] work_buffer;
}

IterFromIndividualLength::IterFromIndividualLength(const std::vector<double>& X,
                                                   const std::vector<size_t>& lengths) {
  maxt = lengths.size();
  n_observations = X.size();
  if (maxt > 0) {
    start = new size_t[maxt]{0};
    end = new size_t[maxt]{0};
    /* cumulated sum */
    end[0] = lengths[0];
    for (size_t i = 1; i < maxt; i++) {
      end[i] = lengths[i] + end[i - 1];
      start[i] = end[i - 1]; 
    }
    if (end[maxt - 1] > n_observations) {
      throw std::invalid_argument("more element in lengths.");
    }
  }
}

size_t IterFromIndividualLength::get_start() {
  if (maxt == 0) return 0;
  if (t <= maxt) {
    return start[t];
  } else {
    return -1;
  }
}

size_t IterFromIndividualLength::get_end() {
  if (maxt == 0) return n_observations;
  if (t <= maxt) {
    return end[t++];
  } else {
    return -1;
  }
}

IterFromIndividualLength::~IterFromIndividualLength() {
  if (maxt > 0) {
    delete [] start;
    delete [] end;
  }
}
