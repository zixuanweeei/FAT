#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Dense>
using namespace Eigen;

struct Kmeans {
  ArrayXd *means;
  ArrayXd *vars;
  int max_iter;
  int n_clusters;
  int random_seed;

  /*!
   * \biref Constructor
   */
  Kmeans(int n_clusters, int max_iter, int random_seed = -1) 
    : n_clusters{n_clusters},
      max_iter{max_iter},
      random_seed{random_seed} {};

  /*!
   * \brief Fit the one dimensional data.
   */
  void fit(const std::vector<double>& X);

  /*!
   * \brief Initialize the data
   */
  void _init();

  /*!
   * \brief Deconstructor
   */
  ~Kmeans();
};
