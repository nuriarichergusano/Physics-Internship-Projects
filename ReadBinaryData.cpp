#include "ReadBinaryData.h"

#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <iostream>
#include <algorithm>

std::array<std::vector<int64_t>, 3> ReadBinaryData(const std::string &fInput,
                                                   const uint64_t &nr,
                                                   const uint16_t nDer,
                                                   const uint16_t nSamples,
                                                   const uint16_t firstSample,
                                                   const uint16_t nChannels)
{
    std::ifstream inFile(fInput, std::ios::binary);
    if (!inFile.is_open())
    {
        throw std::runtime_error("Cannot open file `" + fInput + "`.");
    }

    std::array<std::vector<int64_t>, 3> dataOut;
    std::vector<int64_t> &signalOut{dataOut[0]};
    std::vector<int64_t> &eneOut{dataOut[1]};
    std::vector<int64_t> &idsOut{dataOut[2]};
    signalOut.resize(nSamples * nr * nDer);
    eneOut.resize(nr);
    idsOut.resize(nr);

    uint16_t nsamples = 0;
    uint16_t id = 0;
    float energy = 0;
    std::vector<int16_t> samples;

    uint64_t counter{0};
    while (inFile && counter < nr)
    {
        inFile.read(reinterpret_cast<char *>(&nsamples), sizeof(nsamples));
        samples.resize(nsamples);
        inFile.read(reinterpret_cast<char *>(&id), sizeof(id));
        inFile.read(reinterpret_cast<char *>(&energy), sizeof(energy));
        inFile.read(reinterpret_cast<char *>(&samples[0]), nsamples * sizeof(int16_t));

        for (unsigned int j{0}; j < nSamples; ++j)
        {   
            signalOut[counter * nDer * nSamples + 0 * nSamples + j] = samples[j + firstSample];
        }

        for (unsigned int j{1}; j < nDer; ++j)
        {
            for (unsigned int k{j}; k < nSamples; ++k)
            {
                signalOut[counter * nDer * nSamples + j * nSamples + k] = signalOut[counter * nDer * nSamples + (j - 1) * nSamples + (k)] - signalOut[counter * nDer * nSamples + (j - 1) * nSamples + (k - 1)];
            }
            for (unsigned int k{0}; k < j; ++k)
            {
                signalOut[counter * nDer * nSamples + j * nSamples + k] = signalOut[counter * nDer * nSamples + j * nSamples + j];
            }
        }
        eneOut[counter] = energy;
        idsOut[counter] = id;

        counter++;
    }
    if(counter < nr)
    {
        throw std::runtime_error("File `" + fInput + "` has less data than requested.");
    }

    return dataOut;
}