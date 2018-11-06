#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include "../include/GaussianHMM.h"
using namespace Eigen;

int main() {
  std::cout << "Start ...\n";
  double var = 1.0;
  std::random_device seed;
  std::mt19937 random_number_generator(seed());
  std::normal_distribution<double> white_noise(0.0, var);
  std::cout << "Builded random number generator ...\n";

  ArrayXd clusters(2);
  ArrayXd start(2);
  ArrayXXd trans(2, 2);
  std::cout << "Initialize arrays ...\n";
  clusters << 1, 10;
  start << 0, 1;
  trans << 0.7, 0.3, 0.5, 0.5;
  std::cout << "Finished initialization ...\n";

  std::cout << "Building a hmm object ...\n";
  GaussianHMM hmm(2, clusters, var, 47, 10);
  std::cout << "Finish GaussianHMM.\n";
  std::cout << "GaussianHMM means prior:\n";
  std::cout << hmm.means_prior << "\n";
  *(hmm.pi) = start;
  std::cout << "===== Finished pi =====\n";
  *(hmm.means_) = clusters;
  std::cout << "===== Finished means =====\n";
  *(hmm.A) = trans;
  std::cout << "===== Finished trans =====\n";
  *(hmm.covars_) = ArrayXd::Ones(2) * var;
  std::cout << "===== Finished covars =====\n";
  std::cout << "Finished fit.\n";

  int n_sample = 1000;
  ArrayXi state_seqence(n_sample);
  std::vector<double> X;
  std::cout << "Start to generate sequence ...\n";
  hmm.sample(n_sample, 100, X, state_seqence);
  std::cout << "Finished " << X.size() << " samples generated.\n";

  std::ofstream signal_writer("signal.dat", std::ios_base::out | std::ios_base::trunc);
  signal_writer << "state\tobservation\n";
  for (size_t i = 0; i < X.size(); i++) {
    signal_writer << state_seqence(i) << '\t' << X[i] << '\n';
  }
  signal_writer.close();

  return 0;
}