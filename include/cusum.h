#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

template <typename T>
class Cusum {
public:
  static void cusum(std::vector<T> const& X, double delta, double h, std::vector<T>& mc, std::vector<int>& durations);
};

template <typename T>
void Cusum<T>::cusum(std::vector<T> const& X, double delta, double h, std::vector<T>& mc, std::vector<int>& durations) {
  int Nd = 0;      // detection number
  const int len = X.size();    
  std::vector<int> kd;    // detection time (in samples)
  std::vector<int> krmv;      // estimated change time (in samples)
  int k0 = 0;               // initial sample
  int k = 0;                // current sample
  std::vector<T> m;  m.reserve(5000);  m.emplace_back(X[k0]);  // mean value estimation
  std::vector<T> v;  v.reserve(5000);  v.emplace_back(0);      // variance estimation
  std::vector<T> sp; sp.reserve(5000); sp.emplace_back(0);     // instantaneous log-likelihood ratio for positive jumps
  std::vector<T> Sp; Sp.reserve(5000); Sp.emplace_back(0);     // cumulated sum for positive
  std::vector<T> gp; gp.reserve(5000); gp.emplace_back(0);     // decision function for positive jumps
  std::vector<T> sn; sn.reserve(5000); sn.emplace_back(0);     // instantaneous log-likelihood ratio for negative jumps
  std::vector<T> Sn; Sn.reserve(5000); Sn.emplace_back(0);     // cumulated sum for negative jumps
  std::vector<T> gn; gn.reserve(5000); gn.emplace_back(0);     // decision function for negetive jumps
  
  float prev_delta, post_delta;
  size_t kmin;
  while (k < len - 1) {
    k++;
    prev_delta = X[k] - m[k - 1];                               // online average
    m.emplace_back(m[k - 1] + prev_delta / (k - k0 + 1));
    post_delta = X[k] - m[k];                                   // online s.t.d
    v.emplace_back(std::sqrt(v[k - 1]*v[k - 1] + prev_delta*post_delta));

    // instantaneous log-likelihood ratios
    sp.emplace_back(delta / v[k] * (X[k] - m[k] - delta / 2));
    sn.emplace_back(-delta / v[k] * (X[k] - m[k] + delta / 2));

    // cumulated sums
    Sp.emplace_back(Sp[k - 1] + sp[k]);
    Sn.emplace_back(Sn[k - 1] + sn[k]);

    // decison functions
#ifdef max   // defined in windows.h
    gp.emplace_back(max(gp[k - 1] + sp[k], static_cast<float>(0.0)));
    gn.emplace_back(max(gn[k - 1] + sn[k], static_cast<float>(0.0)));
#else
    gp.emplace_back(std::max(gp[k - 1] + sp[k], static_cast<float>(0.0)));
    gn.emplace_back(std::max(gn[k - 1] + sn[k], static_cast<float>(0.0)));
#endif // max

    // abrupt change detection test
    if (gp[k] > h || gn[k] > h) {
      // detection number and detection time update
      Nd++;
      kd.emplace_back(k);

      // change time estimation
      if (gp[k] > h) {
        kmin = std::min_element(Sp.begin() + k0, Sp.begin() + k + 1) - Sp.begin();
        krmv.emplace_back(kmin - 1);
      } else {
        kmin = std::min_element(Sn.begin() + k0, Sn.begin() + k + 1) - Sn.begin();
        krmv.emplace_back(kmin - 1);
      }

      // algorithm reinitialization
      k0 = k;
      m[k0] = X[k0];
      v[k0] = 0;
      sp[k0] = 0; Sp[k0] = 0; gp[k0] = 0;
      sn[k0] = 0; Sn[k0] = 0; gn[k0] = 0;
    }
  }

  // Piecewise constant segmented signal
  if (Nd == 0)
    mc.emplace_back(std::accumulate(X.begin(), X.end(), 0.0) / len);
  else if (Nd == 1) {
    mc.emplace_back(m[krmv[0]]);
    mc.emplace_back(m[k]);
  }
  else {
    durations.emplace_back(krmv[0]);
    mc.emplace_back(m[krmv[0]]);
    for (int i = 1; i < Nd; i++) {
      durations.emplace_back(krmv[i] - krmv[i - 1]);
      mc.emplace_back(m[krmv[i]]);
    }
    durations.emplace_back(len - krmv[Nd - 1]);
    mc.emplace_back(m[k - 1]);
  }
}
