#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <ctime>
#include <iterator>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include "../include/abfreader.h"
#include "../include/cusum.h"

namespace po = boost::program_options;
namespace file = boost::filesystem;

int main(int argc, char *argv[]) {
  int begin, end, peroid;
  float sigma, threshold;
  std::string filename, destination;
  try {
    po::options_description desc("Fluctuation Analysis");
    desc.add_options()
      ("help", "produce help message")
      ("abf-filename,a", po::value<std::string>(), "The abf file to be processed.")
      ("begin,b", po::value<int>(), "The first point of the interesting segment.")
      ("end,e", po::value<int>(), "The end point of the interesting segment.")
      ("peroid,p", po::value<int>()->default_value(1000000), "Subsegment duration in point.")
      ("destination,d", po::value<std::string>()->default_value("./result"), "The directory to store the result.")
      ("sigma,s", po::value<float>()->default_value(100.), "The parameter SIGMA for cusum algorithm.")
      ("threshold,h", po::value<float>(), "Detection thershold. This can be derived using 2.5*SIGMA.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    // parse --help
    if (vm.count("help")) {
      std::cout << desc << "\n"
                << "Typical usage:\n"
                << "  scatter -a file_to_an.abf -b 100 -e 8000 -p 100000\n"
                << "  scatter --abf-file file_to_an.abf -b 9000 -e 100000 -p 10 --sigma 50\n";
      return 1;
    }
    // parse --abf-filename
    if (vm.count("abf-filename")) {
      filename = vm["abf-filename"].as<std::string>();
      std::cout << "Loads " << filename << " into the memory.\n";
    } else {
      std::cout << "You should specify a abf file.\n";
    }
    // parse --begin and --end
    if (vm.count("begin") && vm.count("end")) {
      begin = vm["begin"].as<int>();
      end = vm["end"].as<int>();
      if (end <= begin) {
        std::cout << "The end of the segment must be greater than the begin. Please check it!\n";
      }
      std::cout << "Extracts the segment between " << begin << " and " << end << ".\n";
    } else {
      std::cout << "Points of begin and end for segment must be set.\n";
    }
    // parse --peroid
    if (vm.count("peroid")) {
      peroid = vm["peroid"].as<int>();
      std::cout << "Subsegment's duration has set to be " << peroid << ".\n";
    }
    // parse --destination
    if (vm.count("destination")) {
      destination = vm["destination"].as<std::string>();
      std::cout << "The result will be stored in " << destination << ".\n";
    }
    // parse --sigma
    if (vm.count("sigma")) {
      sigma = vm["sigma"].as<float>();
      std::cout << "SIGMA = " << sigma;
    }
    // parse --threshold
    if (vm.count("threshold")) {
      threshold = vm["threshold"].as<float>();
      std::cout << ", h(threshold, User Specified) = " << threshold << ".\n";
    } else {
      threshold = 2.5 * sigma;                            // default value
      std::cout << ", h(threshold) = " << threshold << ".\n";
    }
  }
  catch (std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  // const char *f = "E:/nanopore/binbin/cplus/build/17n24017.abf";
  Abfreader reader(filename);
  std::cout << "Number of samples: " << reader.get_num_samples() << std::endl;
  reader.read();
  std::vector<float> segment = reader.read_segment(begin, end);
  std::cout << "Segment size: " << segment.size() << std::endl;

  // apply cusum algorithm
  std::vector<float> mc;                            // stores the average of current amplitude of level segments.
  std::vector<int> durations;                       // stores the dutaitons of the level segments. It matches the sequences stored in the `mc'.
  int num_segment = segment.size() / peroid + 1;    // determines the number of slices.
  std::vector<float>::const_iterator _begin;
  std::vector<float>::const_iterator _end;

  file::path current_path = file::current_path();
  file::path absolute_path = current_path / destination;
  // check whether the directory exists and create it.
  if (!file::exists(absolute_path)) {
    file::create_directory(absolute_path);
    std::cout << "Creates the directory " << absolute_path << "\n.";
  }

  // sliding window
  for (int i = 0; i < num_segment; i++) {
    _begin = segment.begin() + i*peroid;
    _end = (i < num_segment - 1) ? (segment.begin() + (i + 1)*peroid) : segment.end();   // check the bounds of the array
    std::vector<float> subsegment(_begin, _end);                                         // extracts the subsegments in the sliding window
    
    std::clock_t t_start = clock();
    if (mc.empty() && durations.empty())
      Cusum<float>::cusum(subsegment, sigma, threshold, mc, durations);
    else {
      mc.clear(); durations.clear();
      Cusum<float>::cusum(subsegment, sigma, threshold, mc, durations);
    }
    std::cout << "\n" << i << "-th Time taken: " << static_cast<float>(clock() - t_start) / CLOCKS_PER_SEC << "\n";

    // write out the segmented signal
    std::string filename = "segment" + std::to_string(i) + ".csv";
    std::ofstream segment_writer((absolute_path/filename).string(), std::ios_base::out | std::ios_base::trunc);
    if (segment_writer.is_open()) {
      segment_writer << "level,durations\n";
      for (size_t i = 0; i < durations.size(); i++) {
        segment_writer << mc[i] << "," << durations[i] << std::endl;
      }
    }
    segment_writer.close();
    mc.clear(); durations.clear();
  }

  return 0;
}