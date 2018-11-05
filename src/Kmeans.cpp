#include "../include/Kmeans.h"
#include <random>
#include <cmath>

void Kmeans::fit(const std::vector<double>& X) {
  static std::random_device seed;
  static std::mt19937 random_number_generator(random_seed == -1 ? 
                                              seed() : random_seed);
  std::uniform_int_distribution<size_t> indices(0, X.size() - 1);

  _init();
  for (size_t cluster = 0; cluster < n_clusters; cluster++) {
    (*means)(cluster) = X[indices(random_number_generator)];
  }

  std::vector<size_t> assignments(X.size());
  for (size_t i = 0; i < max_iter; i++) {
    for (size_t point = 0; point < X.size(); point++) {
      double best_distance = std::numeric_limits<double>::max();
      size_t best_cluster_belong = 0;
      for (size_t cluster = 0; cluster < n_clusters; cluster++) {
        const double distance = pow(X[point] - (*means)(cluster), 2.0);
      
        if (distance < best_distance) {
          best_distance = distance;
          best_cluster_belong = cluster;
        }
      }
      assignments[point] = best_cluster_belong;
    }

    ArrayXd new_means = ArrayXd::Zero(n_clusters);
    std::vector<size_t> counts(n_clusters, 0);
    for (size_t point = 0; point < X.size(); point++) {
      const size_t cluster = assignments[point];
      new_means(cluster) += X[point];
      counts[cluster] += 1;
    }

    for (size_t cluster = 0; cluster < n_clusters; cluster++) {
      const size_t count = std::max<size_t>(1, counts[cluster]);
      (*means)(cluster) = new_means(cluster) / count;
    }
  }
  std::vector<size_t> counts(n_clusters, 0);
  ArrayXd _vars = ArrayXd::Zero(n_clusters);
  for (size_t point = 0; point < X.size(); point++) {
    const size_t cluster = assignments[point];
    counts[cluster] += 1;
    _vars(cluster) += pow(X[point] - (*means)(cluster), 2.0);
  }
  for (size_t cluster = 0; cluster < n_clusters; cluster++) {
    const size_t count = std::max<size_t>(1, counts[cluster] - 1);
    (*vars)(cluster) = _vars(cluster) / count;
  }
}

void Kmeans::_init() {
  means = new ArrayXd(n_clusters);
  vars = new ArrayXd(n_clusters);
}

Kmeans::~Kmeans() {
  delete means;
  delete vars;
}
