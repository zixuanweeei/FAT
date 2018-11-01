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
  /*!
   * Constructor
   * \param filename The path to the ABF file.
   */
  Abfreader(const char *filename);
  Abfreader(const std::string filename);

  /*!
   * \brief Readout the data from the first channel of the abf.
   */
  void read();

  /*!
   * \brief Read segment sliced from the overall data.
   * \param left The begin point of the data to be sliced.
   * \param right The end point of the data to be sliced.
   */
  std::vector<float> read_segment(int left, int right);

  /*!
   * \brief Get the number of samples in the abf file.
            Because the signals are recorded in two channel, 
            the number of samples is double times the sampling
            steps. The funtion returns the steps as we only use
            the data from the current recording channel.
   */
  int get_num_samples();

  /*!
   * \brief Deconstructor for free space of ADC_data.
   */
  ~Abfreader();
};
