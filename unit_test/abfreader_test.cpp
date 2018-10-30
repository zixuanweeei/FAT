#include <iostream>
#include <vector>
#include "../include/abfreader.h"

int main() {
  Abfreader reader("E:/nanopore/binbin/cplus/build/17n24017.abf");
  int num_samples = reader.get_num_samples();
  std::cout << "Number of samples: " << num_samples << "\n";

  system("pause");
  return 0;
}