#include "ReadRawData.h"


uint64_t swapLong(uint64_t X)
{
    uint64_t x = X;
    x = (x & 0x00000000FFFFFFFF) << 32 | (x & 0xFFFFFFFF00000000) >> 32;
    x = (x & 0x0000FFFF0000FFFF) << 16 | (x & 0xFFFF0000FFFF0000) >> 16;
    x = (x & 0x00FF00FF00FF00FF) << 8 | (x & 0xFF00FF00FF00FF00) >> 8;
    return x;
}

void printHeader(uint64_t *buff)
{
    std::bitset<64> enableChan;
    uint64_t mask00_31 = swapLong(buff[1]);
    uint64_t mask32_63 = swapLong(buff[2]);
    uint64_t mask00_63 = (mask32_63 << 32) | (mask00_31 & 0xFFFFFFFF);
    enableChan = mask00_63;
    for (uint16_t ch = 0; ch < 64; ++ch)
        std::cout << "Channel " << ch << " is: " << ((enableChan.test(ch)) ? "enabled" : "disabled") << std::endl;
    std::cout << "In total " << enableChan.count() << " channels are enabled" << std::endl;
}


Data ProcessBlockOld(const void *input_buffer, int input_size, uint16_t brd){

    uint32_t *inpBuffer = (uint32_t *)input_buffer;
    uint32_t inpBuffSize = input_size / sizeof(uint32_t);

    //  std::cout << "Receiving data in  " << __FILE__ << std::endl;
    int error_code = 0;
    uint32_t posBuffer = 0;
    std::bitset<32> decodeField;
    int countEvents = 0;
    // uint32_t outEvtLen = (sizeof(agataKey) + sizeof(subDataPHA_t));
    //  Aggregate buffer has a header containing different information
    uint32_t aggLength = inpBuffer[posBuffer++] & 0x0FFFFFFF;
    decodeField = inpBuffer[posBuffer++];
    uint32_t board = project_range<27, 31>(decodeField).to_ulong();
    bool boardFailFlag = decodeField.test(26);
    std::bitset<8> channelMask = project_range<0, 7>(decodeField).to_ulong();
    uint32_t aggregateCounter = inpBuffer[posBuffer++] & 0x7FFFFF;
    uint32_t aggregateTimeTag = inpBuffer[posBuffer++];
    uint8_t couples[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    // we need to reconstruct the channel number from the couple id for the
    // 16 channels boards
    int index = 0;
    for (uint16_t bit = 0; bit < 8; bit++)
    {
        if (channelMask.test(bit))
        {
            couples[index] = bit;
            index++;
        }
    }

    uint32_t coupleAggregateSize = 0;
    uint64_t tstamp_10ns = 0;
    uint64_t tstamp = 0;

    uint32_t fineTS = 0;
    int offsetTS = 0;
    uint32_t flags = 0;
    bool pur = false;
    bool satu = false;
    bool lost = false;
    bool trap_sat = false;
    uint32_t chanNum = 0;
    float cfd = 0.;
    // here starts the real decoding of the data for the channels
    Data dataOut;
    for (uint8_t cpl = 0; cpl < channelMask.count(); cpl++)
    {

        if (posBuffer >= aggLength)
            break;

        coupleAggregateSize = inpBuffer[posBuffer++] & 0x3FFFFF;
        dataFormatPHA dataForm;
        dataForm.SetDataFormat(inpBuffer[posBuffer++]);
        uint16_t fNsPerTSample = dataForm.enaDT() ? 8 : 4;
        uint16_t fNsPerTStamp = 4;
        //if (dataForm.evtSize() != 2503)
        //    dataForm.Print();
        const uint16_t evtSize = dataForm.evtSize();
        if (evtSize == 0)
        {
            dataForm.Print();
            break;
        }

        for (uint16_t evt = 0; evt < (coupleAggregateSize - 2) / evtSize; evt++)
        {
            float trig = 0.;
            cfd = 0.;
            decodeField = inpBuffer[posBuffer++];
            tstamp = project_range<0, 30>(decodeField).to_ulong();
            dataOut.ch.push_back(static_cast<uint32_t>(project_range<31, 31>(decodeField).to_ulong() + couples[cpl] * 2));
            dataOut.samples.emplace_back();
            if (dataForm.enaTrace())
            {
                //std::valarray<float> samples;

                if(!dataForm.enaDT())
                    dataOut.samples.back().resize(dataForm.numSamples());
                else
                    dataOut.samples.back().resize(dataForm.numSamples() / 2);

                for (uint16_t s = 0; s < dataForm.numSamples() / 2; ++s)
                {
                    decodeField = inpBuffer[posBuffer++];
                    if (!dataForm.enaDT()){
                        //samples[s * 2] = project_range<0, 13>(decodeField).to_ulong() + (rand() % 1000) / 1000. - 0.5;
                        //samples[s * 2 + 1] = project_range<16, 29>(decodeField).to_ulong() + (rand() % 1000) / 1000. - 0.5;
                        dataOut.samples.back()[s * 2] = project_range<0, 13>(decodeField).to_ulong();
                        dataOut.samples.back()[s * 2 + 1] = project_range<16, 29>(decodeField).to_ulong();
                    }else{
                        dataOut.samples.back()[s] = project_range<16, 29>(decodeField).to_ulong();
                        //dataOut.samples.back()[s] = project_range<0, 13>(decodeField).to_ulong();
                    }
                }
            }

            //	std::cout << "trig = " << trig << " otrig = " << otrig << std::endl;

            if (dataForm.enaExtras2())
            { // decoding of EXTRAS2
                decodeField = inpBuffer[posBuffer++];
                switch (dataForm.confExtras())
                {
                case 0: // Extended Time Stamp [31:16] ; baseline*4 [15:0]
                    tstamp = (static_cast<uint64_t>(project_range<16, 31>(decodeField).to_ulong()) << 31) | (uint64_t)tstamp;
                    break;
                case 1: // Reserved??
                    break;
                case 2: // Extended Time stamp [31:16] ; Reserved [15:10] ; Fine Time Stamp [9:0]
                    fineTS = static_cast<uint16_t>(project_range<0, 9>(decodeField).to_ulong());
                    cfd = fNsPerTStamp / 1.024 * fineTS; // to have it in ps units
                    tstamp = (static_cast<uint64_t>(project_range<16, 31>(decodeField).to_ulong()) << 31) | (uint64_t)tstamp;
                    break;
                case 3: // Reserved
                    break;
                case 4: // Lost Trigger Counter [31:16] ; Total Trigger [15:0]
                    break;
                case 5: // Positive zero crossing [31:16] ; Negative zero crossing [15:0]
                    break;
                case 7: // Reserved
                    break;
                default:
                    break;
                }
            }
            decodeField = inpBuffer[posBuffer++];
            // the last word contains energy[0-14], PU [15] and extra[16:25]
            satu = decodeField.test(4 + 16);
            lost = decodeField.test(5 + 16);
            pur = decodeField.test(9 + 16);
            trap_sat = decodeField.test(10 + 16);

            //	statsData.time[cpl] = tstamp*2/pow( 10, 9 );   // must be in ns

            // if(tsOffset.find(chanNum)!=tsOffset.end())
            tstamp_10ns = tstamp * fNsPerTStamp / 10;
            cfd += (tstamp * fNsPerTStamp - tstamp_10ns * 10) * 1000; // in ps no to loose precision when using uint16_t
            // tstamp_10ns            += offsetTS + fGlobalTSOffset;
            dataOut.TS.push_back(tstamp_10ns);
            dataOut.cfd.push_back(cfd);
            dataOut.brd.push_back(board);

            dataOut.ene.push_back(static_cast<uint16_t>(project_range<0, 15>(decodeField).to_ulong()));

            decodeField = (board << 8 | chanNum);

            // N is set from bits[17:16] of register 0x1nA0 (default value 1024)
            // Adding flags to the evt num
        }
    }
    return dataOut;
}

Data ProcessBlockNew(const void *input_buffer, int n_words, uint16_t brd)
{
    int posBuffer = 0;
    float fineTS_2_GTS = 8 / 1024.;
    const int verbose = 0;
    const bool graphtrace = false;
    const bool anatrace = false;
    uint16_t fineTS;
    uint64_t tstamp;
    bool glbTrig = false;
    Data dataOut;
    while (posBuffer < n_words - 2)
    {
        if (posBuffer > 0)
            posBuffer++;
        if (posBuffer >= n_words - 2)
        {
            // std::cout << "posBuffer = " << posBuffer << " " << n_words << std::endl;
            break;
        }
        std::bitset<64> decodeField = swapLong(((uint64_t *)input_buffer)[posBuffer]);
        dataOut.ch.push_back(project_range<56, 62>(decodeField).to_ulong());
        dataOut.brd.push_back(brd);
        tstamp = project_range<0, 47>(decodeField).to_ulong();
        if (decodeField.test(55))
            std::cout << "Event with statistic only" << std::endl;
        if (verbose > 20)
            std::cout << " Decode field " << std::hex << decodeField.to_ulong() << std::dec << std::endl;
        ////if (verbose > 10)
        ////    std::cout << " --> Event from channel " << lastEvt.ch << " with TStamp = " << lastEvt.TS << std::endl;
        decodeField = swapLong(((uint64_t *)input_buffer)[++posBuffer]);
        dataOut.ene.push_back(project_range<0, 15>(decodeField).to_ulong());
        fineTS = project_range<16, 25>(decodeField).to_ulong();
        dataOut.cfd.push_back(fineTS * fineTS_2_GTS);
        dataOut.TS.push_back((tstamp * 8) / 10);
        dataOut.cfd.back() += (tstamp * 8 - dataOut.TS.back() * 10);
        bool slfTrig = decodeField.test(56);
        bool sftTrig = decodeField.test(55);
        glbTrig = decodeField.test(54);
        bool extTrig = decodeField.test(53);
        if (!slfTrig && !sftTrig && !glbTrig && !extTrig)
            printf("not trigger?? %llx\n", decodeField.to_ulong());
        if (verbose > 10)
            std::cout << "Trigger from " << slfTrig << " " << sftTrig << " " << glbTrig << " " << extTrig << std::endl;
        bool wavePresent = decodeField.test(62);
        if (verbose > 10)
            std::cout << "waves are present? " << ((wavePresent) ? "yes" : "no") << std::endl;
        ////if (verbose > 20)
        ////    std::cout << std::dec << "Energy = " << lastEvt.ene << " fineTS = " << fineTS << std::endl;
        // extra
        bool lastExtra = decodeField.test(63);
        uint16_t extraType = 0;
        uint16_t ap1_Type, ap1_Mult, ap2_Type, ap2_Mult;
        bool ap1_Signed, ap2_Signed;
        uint16_t dp_Type[4];
        uint16_t triggerThres;
        uint16_t downSamp;
        uint64_t deadTime;
        uint32_t savedCount;
        uint32_t triggCount;
        while (!lastExtra)
        {
            if (posBuffer >= n_words - 1)
            {
                std::cout << "breaking because " << posBuffer << " < " << n_words << std::endl;
                break;
            }
            decodeField = swapLong(((uint64_t *)input_buffer)[++posBuffer]);
            lastExtra = decodeField.test(63);
            extraType = project_range<60, 62>(decodeField).to_ulong();
            if (verbose > 5)
                std::cout << "extra type = " << extraType << " at posBuffer = " << posBuffer << std::endl;

            switch (extraType)
            {
            case 0: // waveform configuration

                ap1_Type = project_range<0, 2>(decodeField).to_ulong();
                ap1_Signed = decodeField.test(3);
                ap1_Mult = project_range<4, 5>(decodeField).to_ulong();
                ap2_Type = project_range<6, 8>(decodeField).to_ulong();
                ap2_Signed = decodeField.test(9);
                ap2_Mult = project_range<10, 11>(decodeField).to_ulong();
                if (verbose > 3000)
                    std::cout << "A.P. 1: " << ap1_Type << " " << ap1_Signed << " " << ap1_Mult
                              << "A.P. 2: " << ap2_Type << " " << ap2_Signed << " " << ap2_Mult << std::endl;

                dp_Type[0] = project_range<12, 15>(decodeField).to_ulong();
                dp_Type[1] = project_range<16, 19>(decodeField).to_ulong();
                dp_Type[2] = project_range<20, 23>(decodeField).to_ulong();
                dp_Type[3] = project_range<24, 27>(decodeField).to_ulong();
                if (verbose > 3000)
                {
                    for (uint16_t j = 0; j < 4; ++j)
                    {
                        std::cout << "D.P. " << j << ": " << dp_Type[j] << " ";
                    }
                    std::cout << std::endl;
                }
                triggerThres = project_range<28, 43>(decodeField).to_ulong();
                downSamp = project_range<44, 45>(decodeField).to_ulong();
                if (verbose > 3000)
                    std::cout << "Trigger thres = " << triggerThres << " downSamp = " << downSamp << std::endl;
                break;
            case 1: // dead time
                deadTime = project_range<0, 47>(decodeField).to_ulong();
                break;
            case 2: // counter
                savedCount = project_range<0, 23>(decodeField).to_ulong();
                triggCount = project_range<24, 47>(decodeField).to_ulong();
                break;
            default:
                std::cerr << __LINE__ << ": Type not implemented yet: " << extraType << std::endl;
                break;
            }
            // posBuffer++;
        }
        dataOut.samples.emplace_back();
        if (wavePresent)
        { // if the waves are not present the event stops here
            decodeField = swapLong(((uint64_t *)input_buffer)[++posBuffer]);
            // This is the number of 64 bit words ... the trace will be twice as long;
            uint16_t nbSamples_word = project_range<0, 11>(decodeField).to_ulong();
            ////if (nbSamples_word > 100 && slfTrig)
            ////    std::cout << "Samples to be read: " << nbSamples_word << " for board " << lastEvt.brd << std::endl;
            dataOut.samples.back().resize(nbSamples_word * 2);
            for (uint16_t samp = 0; samp < nbSamples_word; ++samp)
            {
                decodeField = swapLong(((uint64_t *)input_buffer)[++posBuffer]);
                std::bitset<64> sample[2];
                sample[0] = project_range<0, 31>(decodeField);
                sample[1] = project_range<32, 63>(decodeField);
                for (uint16_t s = 0; s < 2; ++s)
                {
                    dataOut.samples.back()[2*samp +s] = project_range<0, 13>(sample[s]).to_ulong();
                    //lastEvt.samples.push_back(project_range<16, 29>(sample[s]).to_ulong() + (rand() % 1000) / 1000. - 0.5);
                }
            }
        }
    }
    return dataOut;
}

std::array<std::vector<int64_t>, 3> ReadRawData(FILE *fInput,
                                                const uint64_t &nr,
                                                const uint16_t nDer,
                                                const uint16_t nSamples,
                                                const uint16_t firstSample,
                                                const uint16_t nChannels){
    std::cout << "Reading " << nr << " events" << std::endl;

    std::vector<uint16_t> enabledBoards{1};

    char buffer[2048 * 2048];

    // Reading the XDAQ Header
    size_t nread = fread(buffer, sizeof(uint32_t), 1, fInput);
    if (nread != 1)
        throw std::runtime_error("Error");
    
    size_t sizeToRead = ((uint32_t *)buffer)[0] & 0x0FFFFFFF;
    nread = fread(buffer, sizeof(uint32_t), sizeToRead - 1, fInput);
    
    uint16_t nbBoard = ((uint32_t *)buffer)[1];
    std::cout << "Number of board in the stream " << nbBoard << std::endl;
    uint64_t fTSOffet = (uint64_t)(((uint32_t *)buffer)[2]) << 32 | ((uint32_t *)buffer)[3];
    uint32_t epochStart = ((uint32_t *)buffer)[4];
    printf("Epoch start = 0x%08X\n", epochStart);

    // Reading the start run event 4 64-bit words
    //MWD waveAnalyzer;
    //waveAnalyzer.Reset(nSamples, true); // user-provided working buffers
    //waveAnalyzer.nDcwin = nSamples / 5; // the first 20% --> check if this is valid
    //waveAnalyzer.nSmooth = 5;
    //waveAnalyzer.nWidthT = 8;
    //waveAnalyzer.nDelayCFD = 6;
    //waveAnalyzer.fFractCDF = 0.25f;
    //waveAnalyzer.fThreshTFA = 20;
    //waveAnalyzer.fThreshCFD = 0;
    //waveAnalyzer.nMinWidthP = 2;
    //waveAnalyzer.nMinWidthN = 6;

    std::array<std::vector<int64_t>, 3> dataOut;
    std::vector<int64_t> &signalOut{dataOut[0]};
    std::vector<int64_t> &eneOut{dataOut[1]};
    std::vector<int64_t> &idsOut{dataOut[2]};
    signalOut.resize(nSamples * nr * nDer);
    eneOut.resize(nr);
    idsOut.resize(nr);

    int verbose {0};
    uint64_t counter{0};
    while (!feof(fInput))
    {
        nread = fread(buffer, sizeof(uint64_t), 1, fInput);
        if (nread != 1){
            printf("nread != 1 for XDAQ Header");
            continue;
        }
        std::bitset<64> decodeField = ((uint64_t *)buffer)[0];
        uint32_t board = project_range<59, 63>(decodeField).to_ulong();
        uint32_t headXDAQ = project_range<32, 59>(decodeField).to_ulong();

        Data bufferOut;
        if (headXDAQ != 0xabacaba){
            uint32_t sizeToRead_XDAQ = project_range<0, 26>(decodeField).to_ulong() * sizeof(uint32_t);
            nread = fread(buffer + 8, sizeToRead_XDAQ - 8, 1, fInput);
            bufferOut = ProcessBlockOld(buffer, sizeToRead_XDAQ, board);
        }
        else
        {
            if (board != 1 && board != 5 && board != 9)
            {
                printf("Warning board number (%i) is strange: %llx\n", board, decodeField.to_ulong());
            }
            uint32_t sizeToRead_XDAQ = project_range<0, 26>(decodeField).to_ulong() * sizeof(uint32_t);
            if (verbose > 2)
                std::cout << "Buffer from board " << board << std::endl;
            if (verbose > 1)
                printf("pos file = %i\n", ftell(fInput));
            nread = fread(buffer, sizeof(uint64_t), 1, fInput);
            if (nread != 1)
            {
                printf("nread != 1 for AGGREGATE Header");
                continue;
            }
            decodeField = swapLong(((uint64_t *)buffer)[0]);
            uint32_t n_words = project_range<0, 31>(decodeField).to_ulong();
            uint32_t headKey = project_range<60, 63>(decodeField).to_ulong();
            uint32_t aggrNb = project_range<32, 55>(decodeField).to_ulong();
            // printf("headKey = %i, n_words = %i, aggrNb = %i, boardNb = %i\n",headKey,n_words,aggrNb,board);
            if (headKey != 2 && headKey != 3)
            {
                printf("Strange headKey 0x%llx\n", decodeField.to_ulong());
            }
            uint32_t sizeToRead_CAEN = n_words * sizeof(uint64_t);
            if (sizeToRead_XDAQ - sizeof(uint64_t) != sizeToRead_CAEN)
                printf("Size to read XDAQ = %i - size to read CAEN = %i\n", sizeToRead_XDAQ, sizeToRead_CAEN);
            n_words = n_words - 1;
            nread = fread(buffer, sizeof(uint64_t), n_words, fInput);
            if (nread != n_words)
            {
                std::cerr << "Cannot read the full aggregate (nread = "
                          << nread << ", n_words = " << n_words << ") "
                          << decodeField.to_ulong() << std::endl;
            }
            if (n_words == 3 && headKey == 0x3)
            {
                if (verbose > 0)
                    printHeader((uint64_t *)buffer);
                if (verbose > 0)
                    printf("Start event at pos file = %i\n", ftell(fInput));
                continue;
            }
            bufferOut = ProcessBlockNew(buffer, n_words, board);
        }

        //Handle from internal Data to output datao
        for (uint16_t i = 0; i < bufferOut.brd.size(); ++i){
            if(counter == nr) return dataOut;

            if(bufferOut.samples[i].size() == 0 || bufferOut.ene[i] == 0) continue;
            if(std::find(enabledBoards.begin(), enabledBoards.end(), bufferOut.brd[i]) == enabledBoards.end()) continue;
            if(bufferOut.ene[i] < 100) continue;
            if(bufferOut.samples[i].size() < nSamples + firstSample)
                throw std::runtime_error(   "The number of samples asked for is: "+std::to_string(nSamples)
                                            +" but actually the max is: "+std::to_string(bufferOut.samples[i].size())
                                            +" for board: "+std::to_string(bufferOut.brd[i]));
            if(bufferOut.ch[i] >= nChannels)
                throw std::runtime_error(   "The number of channels asked for is: "+std::to_string(nChannels)
                                            +" but actually the max is: "+std::to_string(bufferOut.ch[i])
                                            +" for board: "+std::to_string(bufferOut.brd[i]));

            eneOut[counter] = bufferOut.ene[i];

            idsOut[counter] = std::distance(enabledBoards.begin(), 
                                            std::find(enabledBoards.begin(), enabledBoards.end(), bufferOut.brd[i]))*nChannels 
                            + bufferOut.ch[i];
            
                
            //Baseline subtraction
            float baseline = static_cast<std::valarray<float>>(bufferOut.samples[i][std::slice(0, 20, 1)]).sum() / 20.;
            bufferOut.samples[i] -= baseline;
            //bufferOut.samples[i] *= -1;

            for (unsigned int j{0}; j < nSamples; ++j){
                signalOut[counter * nDer * nSamples + 0 * nSamples + j] =  bufferOut.samples[i][j+firstSample];
            }

            for (unsigned int j{1}; j < nDer; ++j){
                for (unsigned int k{j}; k < nSamples; ++k){
                    signalOut[counter * nDer * nSamples + j * nSamples + k] = signalOut[counter * nDer * nSamples + (j - 1) * nSamples + (k)] - signalOut[counter * nDer * nSamples + (j - 1) * nSamples + (k - 1)];
                }
                for (unsigned int k{0}; k < j; ++k){
                    signalOut[counter * nDer * nSamples + j * nSamples + k] = signalOut[counter * nDer * nSamples + j * nSamples + j];
                }
            }
            counter++;
        }
    }
    throw std::runtime_error("The number of events asked for is: "+std::to_string(nr)+" but actually it is: "+std::to_string(counter));
}
