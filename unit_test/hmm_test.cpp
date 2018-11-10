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

  ArrayXd clusters(2);
  ArrayXd start(2);
  ArrayXXd trans(2, 2);
  clusters << 10, 1;
  start << 1, 0;
  trans << 0.88, 0.12, 0.4, 0.6;

  GaussianHMM hmm(2, var, 47, 1000);
  std::cout << "Max epoch - " << hmm.max_epoch << "\n";
  hmm.pi = start;
  hmm.means_ = clusters;
  hmm.A = trans;
  hmm.covars_ = ArrayXd::Ones(2) * var;

  int n_sample = 1000;
  ArrayXi state_seqence(n_sample);
  std::vector<double> X;
  hmm.sample(n_sample, 100, X, state_seqence);

  std::ofstream signal_writer("signal.dat", std::ios_base::out | std::ios_base::trunc);
  signal_writer << "state\tobservation\n";
  for (size_t i = 0; i < X.size(); i++) {
    signal_writer << state_seqence(i) << '\t' << X[i] << '\n';
  }
  signal_writer.close();

  std::cout << "******************* Fitting test *******************\n";
  GaussianHMM fit_hmm(2, 1e-2, -1, 100);
  std::vector<size_t> lengths;
  std::cout << "Start to fit X ...\n";
  fit_hmm.fit(X, lengths);
  std::cout << "Fitted result:\n"
            << "Mean:\n" << fit_hmm.means_ << "\n"
            << "Var: \n" << fit_hmm.covars_ << "\n"
            << "A: \n" << fit_hmm.A << "\n";

  return 0;
}