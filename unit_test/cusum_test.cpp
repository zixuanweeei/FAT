#include <iostream>
#include <fstream>
#include <random>
#include <numeric>
#include <cassert>
#include <ctime>
#include "../include/cusum.h"

int main() {
  // generate a typical signal
  const int n_steps = 100;
  const int mean = 0;
  const int std = 100;

  std::default_random_engine generator;
  std::normal_distribution<float> noise(mean, std);         // Gaussian Distribution noise with mean = 0 and s.t.d = 100
  std::uniform_int_distribution<> dis(1, 10);                // uniform distribution for generating level

  std::vector<int> durations_expected(100, 0);               // stores the durations for levels squence
  std::vector<int> levels_expected(100, 0);                  // stores levels squence
  for (std::vector<int>::iterator i = durations_expected.begin(), j = levels_expected.begin(); 
      i != durations_expected.end(), j != levels_expected.end(); i++, j++) {
    int number = dis(generator) * 1000;
    *i = number;
    number = dis(generator) * 100;
    *j = number;
  }

  int total_length = std::accumulate(durations_expected.begin(), durations_expected.end(), 0);
  std::vector<float> signal(total_length);
  
  // generate the raw signal
  int sum = 0;
  for (std::vector<int>::const_iterator i = durations_expected.begin(); i != durations_expected.end(); i++) {
    size_t n_step = i - durations_expected.begin();
    for (size_t j = 0; j < static_cast<unsigned>(*i); j++) {
      signal[sum + j] = levels_expected[n_step] + noise(generator);     // add noise to a certain level
    }
    sum += *i;
  }
  
  // write out the raw signal
  std::cout << sum << std::endl;
  std::ofstream signal_writer("signal.dat", std::ios_base::out | std::ios_base::trunc);
  if (signal_writer.is_open()) {
    signal_writer << "amp\n";
    for (std::vector<float>::const_iterator i = signal.begin(); i != signal.end(); i++) {
      signal_writer << *i << '\n';
    }
  }
  signal_writer.close();

  // apply cusum algorithm
  std::vector<float> mc;
  std::vector<int> durations;
  std::clock_t s_start = clock();
  if (mc.empty() && durations.empty())
    Cusum<float>::cusum(signal, 100.0, 250.0, mc, durations);
  else {
    mc.clear(); durations.clear();
    Cusum<float>::cusum(signal, 100.0, 250.0, mc, durations);
  }
  assert(durations.size() == mc.size());
  std::cout << "\n\nTime taken: " << static_cast<float>(clock() - s_start) / CLOCKS_PER_SEC << std::endl;

  // write out the segmented signal
  std::ofstream segment_writer("segment.dat", std::ios_base::out | std::ios_base::trunc);
  if (segment_writer.is_open()) {
    segment_writer << "level,durations\n";
    for (size_t i = 0; i < durations.size(); i++) {
      segment_writer << mc[i] << "," << durations[i] << std::endl;
    }
  }
  segment_writer.close();

  return 0;
}