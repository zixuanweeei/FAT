#include <iostream>
#include <Eigen/Dense>
#include "../include/utils.h"

int main() {
  Eigen::Array<double, 1, -1> a(10);
  double b[] = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
  a << 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.;
  std::cout << "Array []logsumexp(range(10)) = " << logsumexp(b, 10) << "\n";
  std::cout << "Eigen - logsumexp(range(10)) = " << logsumexp(a, 10) << "\n";
  Eigen::ArrayXd a_copy = a;
  normalize(a_copy);
  std::cout << "normalize(a) = \n" << a_copy << "\n";

  Eigen::ArrayXXd A(3, 3);
  A << 1, 2, 3,
       4, 5, 6,
       7, 8, 9;
  Eigen::ArrayXXd A_log = A;
  Eigen::ArrayXXd A_ = A;
  normalize(A_);
  std::cout << "normalize(A(3x3)) = \n";
  std::cout << A_ << "\n";
  log_normalize(A_log);
  std::cout << "log_normalize(A(3x3)) = \n";
  std::cout << A_log << "\n";

  double a_scaler = 2.0, b_scaler = 4.0;
  std::cout << "logaddexp(" << a_scaler << 
    " + " << b_scaler << ") = "
    << logaddexp(a_scaler, b_scaler) << "\n";

  Eigen::ArrayXd means(2);
  means << 0, 5;
  Eigen::ArrayXd covar = Eigen::ArrayXd::Ones(2);
  std::vector<double> X(b, b + 10);
  Eigen::ArrayXXd logprob(X.size(), 2);
  log_univariate_normal_density(X, means, covar, logprob);
  std::cout << "log_univariate_normal_density(X) = \n"
    << logprob << "\n";
  
  std::cout << "==================================Forward - backward Test\n";
  size_t n_observations = 10;
  size_t n_components = 2;
  Eigen::ArrayXd log_start(2);
  Eigen::ArrayXXd log_trans(2, 2);
  log_start << std::log(0.5), std::log(0.5);
  log_trans << std::log(0.5), std::log(0.5),
               std::log(0.5), std::log(0.5);
  Eigen::ArrayXXd alpha(10, 2);
  forward(n_observations, n_components, log_start,
          log_trans, logprob, alpha);
  std::cout << "forward -> alpha: \n"
            << alpha << "\n";
  Eigen::ArrayXXd beta(10, 2);
  backward(n_observations, n_components,
           log_trans, logprob, beta);
  std::cout << "backward -> beta: \n"
            << beta << "\n";

  std::cout << "================================Compute log xi\n";
  Eigen::ArrayXXd log_xi_sum(2, 2);
  log_xi_sum = -INFINITY * Eigen::ArrayXXd::Ones(2, 2);
  compute_log_xi_sum(n_observations, n_components, alpha,
                     log_trans, beta, logprob, log_xi_sum);
  std::cout << "Compute_log_xi_sum: \n"
    << log_xi_sum << "\n";

  return 0;
}