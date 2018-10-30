#include <iostream>
#include "../../ABFFIO/ABFFILES.H"
#pragma comment(lib, "../ABFFIO/ABFFIO.lib")

int main() {
  int hFile;
  int nError = 0;
  ABFFileHeader FH;

  DWORD dwMaxEpi = 0;
  UINT uMaxSamples = UINT_MAX;
  UINT uNumSamples = 0;

  if (ABF_ReadOpen("E:/nanopore/data/A4_GA3/pure/15n07036.abf", &hFile, ABF_DATAFILE, &FH, &uMaxSamples, &dwMaxEpi, &nError)) {
    if (ABF_GetNumSamples(hFile, &FH, 1, &uNumSamples, &nError)) {
      std::cout << "Number of sample: " << uNumSamples << std::endl;
      std::cout << "Number of sweeps: " << dwMaxEpi << std::endl;
      std::cout << "Number of samples per episode: " << FH.lNumSamplesPerEpisode << std::endl;
    }
    else {
      std::cout << nError << std::endl;
    }
    std::cout << "Get number of samples finished." << std::endl;

		// Get first physical channel number and name
		int nFirstPhsicalChannel = FH.nADCSamplingSeq[0];
		char *psSignalName = FH.sADCChannelName[nFirstPhsicalChannel];
		std::cout << "The first acquired channel (" << psSignalName
			<< ") comes from ADC channel " << nFirstPhsicalChannel
			<< ", its units is " << FH.sADCUnits[nFirstPhsicalChannel]
			<< ", version " << FH.fFileVersionNumber
			<< std::endl;

    getchar();
		FLOAT *pfBuffer = new FLOAT[uNumSamples];
		if (ABF_ReadChannel(hFile, &FH, nFirstPhsicalChannel, 0, pfBuffer, &uMaxSamples, &nError))
			for (size_t i = 0; i < 10; i++)
				std::cout << pfBuffer[i] << ' ';
		else
			std::cout << nError << std::endl;
		ABF_Close(hFile, NULL);
		delete[] pfBuffer;
  }
  std::cout << "\nFinished\n";

  system("pause");
  return 0;
}