#include <torch/extension.h>


#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

#include <torch/csrc/Export.h>

#include <torch/extension.h>
#include <torch/data/datasets.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace td = torch::data;


#include <cstddef>
#include <string>

#include "ReadRawData.h"
#include "ReadBinaryData.h"

namespace torch::data::datasets {
/// The PIDDataset dataset.
    class TORCH_API PIDDataset : public Dataset<PIDDataset> {
    public:
    /// The mode in which the dataset is loaded.
    enum class Mode { kTrain, kTest } mode;

    /// Loads the PIDDataset dataset from the `root` path.
    ///
    /// The supplied `root` path should contain the *content* of the unzipped
    /// PIDDataset dataset, available from http://yann.lecun.com/exdb/mnist.
    explicit PIDDataset(const std::string& root, 
                        Mode mode = Mode::kTrain, 
                        const unsigned int& count=10000, 
                        const unsigned int& nDer=1, 
                        const unsigned int& nSamples=128, 
                        const unsigned int& firstSample=0, 
                        const unsigned int& nChannels=64, 
                        const bool& normalized=true,
                        const bool& discardEnergies=false,
                        const double& minEnergy=-1.0,
                        const double& maxEnergy=-1.0);

    /// Returns the `PIDData` at the given `index`.
    Example<> get(size_t index) override;

    /// Returns the size of the dataset.
    optional<size_t> size() const override;

    /// Returns true if this is the training subset of PIDDataset.
    // NOLINTNEXTLINE(bugprone-exception-escape)
    bool is_train() const noexcept;

    /// Returns all images stacked into a single tensor.
    const Tensor& signals() const;

    /// Returns all targets stacked into a single tensor.
    const Tensor& ids() const;

    private:
    Tensor signals_, ids_;
};
} // namespace datasets

// Binding code using PYBIND11_MODULE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
// Define the enum class Mode
py::enum_<td::datasets::PIDDataset::Mode>(m, "PIDDatasetMode")
.value("kTrain", td::datasets::PIDDataset::Mode::kTrain)
.value("kTest", td::datasets::PIDDataset::Mode::kTest)
.export_values();

// Define the class PIDDataset
py::class_<td::datasets::PIDDataset>(m, "PIDDataset")
.def(py::init<  const std::string&, 
                td::datasets::PIDDataset::Mode, 
                const unsigned int&, 
                const unsigned int&, 
                const unsigned int&, 
                const unsigned int&, 
                const unsigned int&, 
                const bool&,
                const bool&,
                const double&,
                const double&
                >(), 
        py::arg("root"), 
        py::arg("mode") = td::datasets::PIDDataset::Mode::kTrain,
        py::arg("count") = 1000, 
        py::arg("nder") = 2, 
        py::arg("nsamples") = 128, 
        py::arg("firstsample") = 0, 
        py::arg("nchannels") = 64, 
        py::arg("normalized") = true,
        py::arg("discardenergies") = false,
        py::arg("minenergy") = -1.0,
        py::arg("maxenergy") = -1.0
)
.def("__getitem__",
    [](td::datasets::PIDDataset& fun, size_t index) {
    auto example = fun.get(index);
    // Convert C++ tensors to PyTorch tensors
    return py::make_tuple(example.data, example.target);})
.def("__len__", &td::datasets::PIDDataset::size)
.def("is_train", &td::datasets::PIDDataset::is_train)
.def("signals", &td::datasets::PIDDataset::signals, py::return_value_policy::reference_internal)
.def("ids", &td::datasets::PIDDataset::ids, py::return_value_policy::reference_internal);
}

namespace torch::data::datasets {
    PIDDataset::PIDDataset( const std::string& file, 
                            Mode mode, 
                            const unsigned int& count, 
                            const unsigned int& nDer, 
                            const unsigned int& nSamples, 
                            const unsigned int& firstSample, 
                            const unsigned int& nChannels, 
                            const bool& normalized,
                            const bool& discardEnergies,
                            const double& minEnergy,
                            const double& maxEnergy
                            ):mode(mode){

        
        //Check if file ends with .root
        std::array<std::vector<int64_t>, 3> data;
        auto& signalsV = data[0];
        auto& energiesV = data[1];
        auto& idsV = data[2];

        if(file.find(".dat") == std::string::npos){
            FILE *fInput = fopen(file.c_str(),"rb");
            if (!fInput) {
                throw std::runtime_error("Cannot open file `" + file + "`.");
            }
            data = ReadRawData(fInput, count, nDer, nSamples, firstSample, nChannels);
            fclose(fInput);
        } else {
            data = ReadBinaryData(file, count, nDer, nSamples, firstSample, nChannels);
        }

        //Not format the tensors correctly
        signals_ = torch::from_blob(static_cast<int64_t*>(signalsV.data()) ,{count, 1, nDer, nSamples}, torch::kInt64)
                .clone().to(torch::kFloat32);
        auto energies_ = torch::from_blob(static_cast<int64_t*>(energiesV.data()), {energiesV.size(),1}, torch::kInt64)
                .clone().to(torch::kFloat32);

        if(normalized){
            signals_.div_(energies_.unsqueeze(-1).unsqueeze(-1));
        }

        signals_.div_(signals_.max());
        energies_.div_(energies_.max());
        
        ids_ = torch::from_blob(static_cast<int64_t*>(idsV.data()), {idsV.size(), 1}, torch::kInt64)
                .clone().to(torch::kInt64);

        if(minEnergy >= 0){
            auto mask = (energies_ >= minEnergy);
            signals_ = signals_.index({mask});
            energies_ = energies_.index({mask});
            ids_ = ids_.index({mask});
        }
        if(maxEnergy >= 0){
            auto mask = (energies_ <= maxEnergy);
            signals_ = signals_.index({mask});
            energies_ = energies_.index({mask});
            ids_ = ids_.index({mask});
        }
        
        energies_ = energies_.unsqueeze(1).unsqueeze(2).expand({-1, nDer, 1});
        
        if (!discardEnergies){
            signals_ = torch::cat({signals_, energies_}, 2);
        }

        signals_ = signals_.unsqueeze(1);
        //if(!discardEnergies) {
        //    //ids_ = torch::nn::functional::one_hot(ids_).to(torch::kFloat32);
        //    //Concatenate the energies to the signals
        //    std::cout << signals_.sizes() << std::endl;
        //    std::cout << energies_.sizes() << std::endl;
        //    signals_ = torch::cat({signals_, energies_}, 1);

        //    //energies_ = torch::cat({energies_, ids_}, 1);
        //} else 
        //    energies_ = ids_;
    }

    Example<> PIDDataset::get(size_t index) {
        return {signals_[index], ids_[index]};
    }

    optional<size_t> PIDDataset::size() const {
        return signals_.size(0);
    }

// NOLINTNEXTLINE(bugprone-exception-escape)
    bool PIDDataset::is_train() const noexcept {
    return mode==Mode::kTrain;
}

const Tensor& PIDDataset::signals() const {
    return signals_;
}

const Tensor& PIDDataset::ids() const {
    return ids_;
}

} // namespace datasets
