#include "../include/abfreader.h"
#include <string>

// user defined default constructor
Abfreader::Abfreader(const char *filename_) {
  filename = filename_;
  DWORD max_episodes = 0;
  UINT max_samples = UINT_MAX;
  UINT num_sample_u = 0;

  if (ABF_ReadOpen(filename.c_str(), &file_handle, ABF_DATAFILE, &file_header, &max_samples, &max_episodes, &error_flag)) {
    if (ABF_GetNumSamples(file_handle, &file_header, 1, &num_sample_u, &error_flag)) {
      if (num_sample_u*2 != file_header.lNumSamplesPerEpisode) {
        std::cout << "Number of samples per episode doesn't match that read by ReadOpen\n";
        throw std::runtime_error("Unsupported acquisition mode.");
      }
      else {
        num_samples = num_sample_u;
      }
    } else {
      std::cout << "Error [" << error_flag << "]\n";
      throw std::runtime_error("Failed to get number of samples.");
    }
  } else {
    std::cout << "Error [" << error_flag << "]\n";
    std::string error_massage("Can't open the abf file: ");
    error_massage += filename;
    throw std::runtime_error(error_massage);
  }
}

// user defined default constructor
Abfreader::Abfreader(const std::string filename_) {
  filename = filename_;
  DWORD max_episodes = 0;
  UINT max_samples = UINT_MAX;
  UINT num_sample_u = 0;

  if (ABF_ReadOpen(filename.c_str(), &file_handle, ABF_DATAFILE, &file_header, &max_samples, &max_episodes, &error_flag)) {
    if (ABF_GetNumSamples(file_handle, &file_header, 1, &num_sample_u, &error_flag)) {
      if (num_sample_u * 2 != file_header.lNumSamplesPerEpisode) {
        std::cout << "Number of samples per episode doesn't match that read by ReadOpen\n";
        throw std::runtime_error("Unsupported acquisition mode.");
      } else {
        num_samples = num_sample_u;
      }
    } else {
      std::cout << "Error [" << error_flag << "]\n";
      throw std::runtime_error("Failed to get number of samples.");
    }
  } else {
    std::cout << "Error [" << error_flag << "]\n";
    std::string error_massage("Can't open the abf file: ");
    error_massage += filename;
    throw std::runtime_error(error_massage);
  }
}

// read all data samples from the file
void Abfreader::read() {
  float *_data_buffer = new float[num_samples];
  int no_first_phsical_channel = file_header.nADCSamplingSeq[0];
  if (ABF_ReadChannel(file_handle, &file_header, no_first_phsical_channel, 1, _data_buffer, &num_samples, &error_flag)) {
    ADC_data = new std::vector<float>(_data_buffer, _data_buffer + num_samples);
    // std::cout << ADC_data->size() << " of samples has put into a vector.\n";
  } else {
    std::cout << "Error [" << error_flag << "]\n";
    throw std::runtime_error("Can't read data from the first channel.");
  }
  delete[] _data_buffer;
}

// data slice
std::vector<float> Abfreader::read_segment(int left, int right) {
  // std::cout << "There are " << ADC_data->size() << " of samples stored.\n";
  std::vector<float> segment(ADC_data->begin() + left, ADC_data->begin() + right);
  //std::cout << segment->size() << " of data sliced.\n";

  return segment;
}

// get number of samples
int Abfreader::get_num_samples() {
  return num_samples;
}

// deconstructor
Abfreader::~Abfreader() {
  delete ADC_data;
}