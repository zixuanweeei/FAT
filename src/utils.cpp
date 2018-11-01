#include "../include/utils.h"
#include <iostream>

bool ConvergenceMonitor::report(double logprob) {
  if (verbose) {
    double delta = history.size() ? (logprob - history.back()) : std::nan("1");
    printf("Epoch[%04d] - logrob:%6.4f, delta:%6.4f", iter, logprob, delta);
  }

  history.push_back(logprob);
  iter++;

  return (iter == max_epoch || 
          (iter > 1 && history[iter - 1] - history[iter - 2] < tol));
}