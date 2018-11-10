#pragma once
#include <Eigen/Dense>
using namespace Eigen;

/*!
 * \brief Some statistics for updating HMM parameters
 */
struct Stats {
  size_t nobs;
  ArrayXd start;
  ArrayXXd trans;

  /* for GaussianHMM */
  ArrayXd post;
  ArrayXd obs;
  ArrayXd obs_square;

  /* Constructor */
  Stats(size_t n_observations, size_t n_components) {
    nobs = 0;
    start = ArrayXd::Zero(n_components);
    trans = ArrayXXd::Zero(n_components, n_components);
    post = ArrayXd::Zero(n_components);
    obs = ArrayXd::Zero(n_components);
    obs_square = ArrayXd::Zero(n_components);
  }
};
