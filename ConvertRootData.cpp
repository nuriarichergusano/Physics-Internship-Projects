#include <TTree.h> 
#include <TFile.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>
#include <TGraph.h>
#include <TCanvas.h>
#include <TH2.h>
#include <TSpline.h>
#include <fstream>
#include <TStyle.h>

#include <iostream>

void ConvertRootData(const std::string& fInput){

    TFile file(fInput.c_str(), "read");
    TTree *tree = (TTree*)file.Get("events");
    if(!tree){
        std::cerr << "Error: Tree not found in file " << fInput << std::endl;
        exit(1);
    }
    TTreeReader     fReader;  //!the tree reader
    TTreeReaderValue<Int_t> run = {fReader, "run"};
    TTreeReaderValue<Int_t> mtot = {fReader, "mtot"};
    TTreeReaderArray<Int_t> idtel = {fReader, "idtel"};
    TTreeReaderArray<Int_t> Blk = {fReader, "Blk"};
    TTreeReaderArray<Int_t> Qua = {fReader, "Qua"};
    TTreeReaderArray<Int_t> Tel = {fReader, "Tel"};
    TTreeReaderArray<Int_t> Z = {fReader, "Z"};
    TTreeReaderArray<Int_t> A = {fReader, "A"};
    //TTreeReaderArray<Float_t> PID = {fReader, "Pid"};
    TTreeReaderArray<Float_t> Zr = {fReader, "Zr"};
    TTreeReaderArray<Float_t> Ar = {fReader, "Ar"};
    TTreeReaderArray<Int_t> Aid = {fReader, "Aid"};
    TTreeReaderArray<Int_t> IDCode = {fReader, "IDCode"};
    TTreeReaderArray<Int_t> ECode = {fReader, "ECode"};
    TTreeReaderArray<Int_t> IDType = {fReader, "IDType"};
    TTreeReaderArray<Float_t> IDY = {fReader, "IDY"};
    TTreeReaderArray<Float_t> IDX = {fReader, "IDX"};
    TTreeReaderArray<Float_t> pippa = {fReader, "pippa"};
    TTreeReaderArray<Float_t> diff_chi = {fReader, "diff_chi"};
    TTreeReaderArray<Float_t> diff_si1 = {fReader, "diff_si1"};
    TTreeReaderArray<Float_t> diff_si2 = {fReader, "diff_si2"};
    TTreeReaderArray<Float_t> diff_csi = {fReader, "diff_csi"};
    TTreeReaderArray<Float_t> Esi1 = {fReader, "Esi1"};
    TTreeReaderArray<Float_t> Esi2 = {fReader, "Esi2"};
    TTreeReaderArray<Float_t> Ecsi = {fReader, "Ecsi"};
    TTreeReaderArray<Float_t> EcsiDaSi = {fReader, "EcsiDaSi"};
    TTreeReaderArray<Float_t> Etot = {fReader, "Etot"};
    TTreeReaderArray<Float_t> vv = {fReader, "vv"};
    TTreeReaderArray<Float_t> vx = {fReader, "vx"};
    TTreeReaderArray<Float_t> vy = {fReader, "vy"};
    TTreeReaderArray<Float_t> vz = {fReader, "vz"};
    TTreeReaderArray<Float_t> vper = {fReader, "vper"};
    TTreeReaderArray<Float_t> CHsi1 = {fReader, "CHsi1"};
    TTreeReaderArray<Float_t> CHsi2 = {fReader, "CHsi2"};
    TTreeReaderArray<Float_t> CHcsi = {fReader, "CHcsi"};
    TTreeReaderArray<Float_t> theta = {fReader, "theta"};
    TTreeReaderArray<Float_t> phi = {fReader, "phi"};
    TTreeReaderArray<Float_t> x = {fReader, "x"};
    TTreeReaderArray<Float_t> y = {fReader, "y"};
    TTreeReaderArray<Int_t> ADCq2_1 = {fReader, "ADCq2_1"};
    TTreeReaderArray<Int_t> ADCq2_2 = {fReader, "ADCq2_2"};
    TTreeReaderArray<Int_t> ADCq2_3 = {fReader, "ADCq2_3"};
    TTreeReaderArray<Int_t> ADCq2_4 = {fReader, "ADCq2_4"};
    TTreeReaderArray<Int_t> ADCq2_5 = {fReader, "ADCq2_5"};
    TTreeReaderArray<Int_t> ADCq2_6 = {fReader, "ADCq2_6"};

    fReader.SetTree(tree);

    //TH2D *h = new TH2D("h", "h", 1000, 0, 1200, 1000, 0, 1200);
    //TH2D *h18 = new TH2D("h18", "h18", 1000, 0, 1200, 1000, 0, 1200);
    
    uint64_t counter{0};
    //binary output file
    std::string outFileName = fInput.substr(0, fInput.find_last_of('.')) + ".dat";
    std::ofstream outFile(outFileName, std::ios::binary);
        
    std::vector<int16_t> samples;

    int nsamplesbaseline = 30;
    int totsamples = 200;

    for(Long64_t entry = 1; entry < tree->GetEntries(); ++entry){
        fReader.SetEntry(entry);
        if(*mtot != 1 || Z[0] == 0 || Esi1[0]<0.1 || Esi2[0]<0.1 || ECode[0] != 0) continue;
        //Create a tgraph of adcq2_1
        //if(*mtot > 0){
        //    h->Fill(Esi1[0], Esi2[0]);
        //    if(Z[0] == 18){
        //        h18->Fill(Esi1[0], Esi2[0]);
        //    }
        //    //for(size_t i = 0; i < ADCq2_1.GetSize(); ++i){
        //    //    h->Fill(i, ADCq2_1[i]);
        //    //}
        //}
    
        //std::cout << x.GetSize() << std::endl;
        //Get the avg og the first 10 samples
        const int nsamplesbaseline = 30;
        //double avg = 0;
        //for(int i = 0; i < nsamplesbaseline; ++i){
        //    avg += ADCq2_1[i];
        //}
        //avg /= nsamplesbaseline;

        //Write the nsamples in outFile
        uint16_t nsamples = ADCq2_1.GetSize();
        samples.resize(nsamples);

        uint16_t id = Z[0]-1;
        float energy = Esi2[0];
        
        auto findMax = [&]() -> std::pair<int, float>{
            auto max = std::numeric_limits<int>::min();
            int maxidx = 0;
            for(int i = 0; i < nsamples-1; ++i){
                if(ADCq2_1[i+1]-ADCq2_1[i] > max){
                    maxidx = i;
                    max = ADCq2_1[i+1]-ADCq2_1[i];
                }
            }
            return {maxidx, max};
        };
        auto [shift, maxval] = findMax();


        if(shift - nsamplesbaseline< 0 || shift - nsamplesbaseline >= nsamples) continue;

        //TGraph g;
        for(int i = 0; i < nsamples; ++i){
            samples[i] = ADCq2_1[shift - nsamplesbaseline + i+1] - ADCq2_1[shift - nsamplesbaseline + i];
            //if(i < 192) g.AddPoint(i, samples.at(i));
        }
        if(maxval > 2000) {
            std::cout << "Maxval: " << maxval << std::endl;
            TGraph g;
            for(int i = 0; i < nsamples; ++i){
                g.AddPoint(i, samples.at(i));
            }
            TCanvas cv;
            cv.Draw();
            g.Draw();
            cv.WaitPrimitive();
        }

        //TCanvas cv;
        //cv.Draw();
        //g.Draw();
        //cv.WaitPrimitive();
        
        outFile.write(reinterpret_cast<const char*>(&nsamples), sizeof(nsamples));
        outFile.write(reinterpret_cast<const char*>(&id), sizeof(id));
        outFile.write(reinterpret_cast<const char*>(&energy), sizeof(energy));
        outFile.write(reinterpret_cast<const char*>(&samples[0]), nsamples * sizeof(int16_t));
        //TGraph *gr1 = new TGraph();
        //for(size_t i = 0; i < ADCq2_1.GetSize(); ++i){
        //    if (i < firstSample) continue;
        //    if (i >= firstSample + nSamples) break;
        //    gr1->AddPoint(i, ADCq2_1[i]-avg);
        //}

        //TCanvas *c = new TCanvas("c", "c", 800, 600);
        //gr1->Draw("AP");

        //c->WaitPrimitive();
    }
    //TCanvas *c = new TCanvas("c", "c", 800, 600);
    //h->Draw("colz");
    //h18->Draw("colz");

    //c->WaitPrimitive();
}