#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <set>
#include <valarray>
#include <array>
#include <stdint.h>
#include <map>
#include <bitset>
#include <cstring>
#include "mwdlib.h"

template <std::size_t R, std::size_t L, std::size_t N>
std::bitset<N> project_range(std::bitset<N> b)
{
    static_assert(R <= L && L <= N, "invalid bitrange");
    b = b << (N - L - 1);       // drop R rightmost bits
    b = b >> (R + (N - L - 1)); // drop L-1 leftmost bits
    return b;
}

struct dataFormatPHA
{
    std::bitset<32> dataFormat;
    void SetDataFormat(uint32_t form) { dataFormat = form; }
    bool enaDT() const { return dataFormat.test(31); }
    bool enaEnergy() const { return dataFormat.test(30); }
    bool enaTS() const { return dataFormat.test(29); }
    bool enaExtras2() const { return dataFormat.test(28); }
    bool enaTrace() const { return dataFormat.test(27); }
    uint8_t confExtras() const
    {
        return static_cast<uint8_t>(
            project_range<24, 26>(dataFormat).to_ulong());
    }
    uint8_t confAnaProbe1() const
    {
        return static_cast<uint8_t>(
            project_range<22, 23>(dataFormat).to_ulong());
    }
    uint8_t confAnaProbe2() const
    {
        return static_cast<uint8_t>(
            project_range<20, 21>(dataFormat).to_ulong());
    }
    uint8_t confDigProbe() const
    {
        return static_cast<uint8_t>(
            project_range<16, 19>(dataFormat).to_ulong());
    }
    uint16_t numSamples() const
    {
        return static_cast<uint16_t>(
                   project_range<0, 15>(dataFormat).to_ulong()) *
               8;
    }
    uint16_t evtSize() const
    {
        uint16_t evtSize = 0;
        // SP
        // if(enaDT()) evtSize+=1;
        // end SP
        if (enaEnergy())
            evtSize += 1;
        if (enaTS())
            evtSize += 1;
        if (enaExtras2())
            evtSize += 1;
        if (enaTrace())
            evtSize += numSamples() / 2;
        return evtSize;
    }
    void Print()
    {
        printf("Data format (%08X): ", dataFormat.to_ulong());
        std::cout << dataFormat << std::endl;
        printf(" --- Dual Trace [31] are %s\n",
               ((enaDT()) ? "enabled" : "disabled"));
        printf(" --- Energy [30] are %s\n",
               ((enaEnergy()) ? "enabled" : "disabled"));
        printf(" --- TimeTag [29] are %s\n",
               ((enaTS()) ? "enabled" : "disabled"));
        printf(" --- Extras [28] are %s\n",
               ((enaExtras2()) ? "enabled" : "disabled"));
        if (enaExtras2())
            printf(" ---- Extras configuration [26:24] set to %i \n",
                   confExtras());
        printf(" --- Trace [27] are %s\n",
               ((enaTrace()) ? "enabled" : "disabled"));
        printf(" --- Number of samples %i\n", numSamples());
        printf(" --- Event length %i 32-bit words\n", evtSize());
    }
};

struct Data
{
    std::vector<uint32_t> brd;
    std::vector<uint32_t> ch;
    std::vector<float> ene;
    std::vector<float> cfd;
    std::vector<uint64_t> TS;
    std::vector<std::valarray<float>> samples;
    void Print()
    {
        for (size_t i = 0; i < brd.size(); ++i)
        {
            std::cout << "Board = " << brd[i] << " Channel = " << ch[i] << " Energy = " << ene[i] << " CFD = " << cfd[i] << " TS = " << TS[i] << std::endl;
        }
    }
};

std::array<std::vector<int64_t>, 3> ReadRawData(FILE *fInput,
                                                const uint64_t &nr,
                                                const uint16_t nDer,
                                                const uint16_t nSamples,
                                                const uint16_t firstSample,
                                                const uint16_t nChannels);