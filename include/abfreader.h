#pragma once
#include <iostream>
#include <vector>
#include <string>
#include "../ABFFIO/ABFFILES.H"
#pragma comment(lib, "../ABFFIO/ABFFIO.lib")

class Abfreader {
private:
  std::string filename;
  UINT num_samples;
  std::vector<float> *ADC_data;

  int file_handle;
  int error_flag = 0;
  ABFFileHeader file_header;
public:
  Abfreader(const char *filename);
  Abfreader(const std::string filename);
  void read();
  std::vector<float> read_segment(int left, int right);
  int get_num_samples();
  ~Abfreader();
};
