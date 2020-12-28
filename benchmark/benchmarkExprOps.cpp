#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <cfloat>
#include <map>
#include <cstring>
#include <cstdlib>
#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <dirent.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include <dlfcn.h>
#include "MNN_generated.h"
#include <MNN/MNNForwardType.h>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "ExprModels.hpp"

#include "core/Backend.hpp"
#include "core/BackendFactory.hpp"
#include "core/Session.hpp"
#include "core/MultiSession.hpp"
#include "core/MultiPipeline.hpp"
#include "core/TensorUtils.hpp"
#include "revertMNNModel.hpp"
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>


using namespace MNN;
using namespace MNN::Express;

struct Model {
    std::string name;
    std::string model_file;
};

#if !defined(_MSC_VER)
inline bool file_exist(const char *file) {
    struct stat buffer;
    return stat(file, &buffer) == 0;
}
#endif

std::vector<Model> findModelFiles(const char *dir) {
    std::vector<Model> models;
#if defined(_MSC_VER)
    WIN32_FIND_DATA ffd;
    HANDLE hFind = INVALID_HANDLE_VALUE;
    std::string mnn_model_pattern = std::string(dir) + "\\*.mnn";
    hFind = FindFirstFile(mnn_model_pattern.c_str(), &ffd);
    if (INVALID_HANDLE_VALUE == hFind) {
        std::cout << "open " << dir << " failed: " << strerror(errno) << std::endl;
        return models;
    }
    do {
        Model m;
        m.name = ffd.cFileName;
        m.model_file = std::string(dir) + "\\" + m.name;
        if (INVALID_FILE_ATTRIBUTES != GetFileAttributes(m.model_file.c_str()) && GetLastError() != ERROR_FILE_NOT_FOUND) {
            models.push_back(std::move(m));
        }
    } while (FindNextFile(hFind, &ffd) != 0);
    FindClose(hFind);
#else
    DIR *root;
    if ((root = opendir(dir)) == NULL) {
        std::cout << "open " << dir << " failed: " << strerror(errno) << std::endl;
        return models;
    }

    struct dirent *ent;
    while ((ent = readdir(root)) != NULL) {
        Model m;
        if (ent->d_name[0] != '.') {
            m.name = ent->d_name;
            m.model_file = std::string(dir) + "/" + m.name;
            if (file_exist(m.model_file.c_str())) {
                models.push_back(std::move(m));
            }
        }
    }
    closedir(root);
#endif
    return models;
}

static inline uint64_t getTimeInUs() {
    uint64_t time;
#if defined(_MSC_VER)
    LARGE_INTEGER now, freq;
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&freq);
    uint64_t sec = now.QuadPart / freq.QuadPart;
    uint64_t usec = (now.QuadPart % freq.QuadPart) * 1000000 / freq.QuadPart;
    time = sec * 1000000 + usec;
#else
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    time = static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
#endif
    return time;
}

void setInputData(MNN::Tensor *tensor) {
    float *data = tensor->host<float>();
    for (int i = 0; i < tensor->elementSize(); i++) {
        data[i] = Revert::getRandValue();
    }
}

void setInputData(MNN::Tensor *tensor, float value) {
    float *data = tensor->host<float>();
    for (int i = 0; i < tensor->elementSize(); i++) {
        data[i] = value;
    }
}

static inline std::string forwardType(MNNForwardType type) {
    switch (type) {
        case MNN_FORWARD_CPU:
            return "CPU";
        case MNN_FORWARD_VULKAN:
            return "Vulkan";
        case MNN_FORWARD_OPENCL:
            return "OpenCL";
        case MNN_FORWARD_METAL:
            return "Metal";
        default:
            break;
    }
    return "N/A";
}

static std::vector<std::string> splitArgs(const std::string& args, const std::string& delimiter) {
    std::vector<std::string> result;
    size_t pos = 0, nextPos = args.find(delimiter, 0);
    while (nextPos != std::string::npos) {
        result.push_back(args.substr(pos, nextPos - pos));
        pos = nextPos + delimiter.length();
        nextPos = args.find(delimiter, pos);
    }
    result.push_back(args.substr(pos, args.length() - pos));
    return result;
}

static void displayStats(const std::string& name, const std::vector<float>& costs) {
    float max = 0, min = FLT_MAX, sum = 0, avg;
    for (auto v : costs) {
        max = max < v ? v : max;
        min = min > v ? v : min;
        sum += v;
    }
    avg = costs.size() > 0 ? sum / costs.size() : 0;
    printf("[ - ] %-24s    max = %8.3fms  min = %8.3fms  avg = %8.3fms\n", name.c_str(), max, avg == 0 ? 0 : min, avg);
}

Interpreter* createFromVARP(VARP netOutput) {
    std::unique_ptr<NetT> netTable(new NetT);
    Variable::save({netOutput}, netTable.get());
    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = CreateNet(builder, netTable.get());
    builder.Finish(offset);
    const void* buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();
    return Interpreter::createFromBuffer(buf, size);
}

Interpreter* createFromModel(Model& model) {
    auto revertor = std::unique_ptr<Revert>(new Revert(model.model_file.c_str()));
    revertor->initialize();
    auto modelBuffer = revertor->getBuffer();
    const auto bufferSize = revertor->getBufferSize();
    return Interpreter::createFromBuffer(modelBuffer, bufferSize);
}


static std::vector<float> runNet(std::vector<std::shared_ptr<Interpreter>> nets, int forward = MNN_FORWARD_CPU, int numberThread = 4, int precision = 2, int batch = 1, int fuseCount = 1) {
    
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)precision;
    backendConfig.power = MNN::BackendConfig::Power_High;

    MNN::Backend::Info info;
    info.type = static_cast<MNNForwardType>(forward);
    info.numThread = numberThread;
    info.user = &backendConfig;

    auto backend = MNN::BackendFactory::create(info);

    MNN::ScheduleConfig config;
    MNN::MultiSession multiSession;
    config.backend = backend;

    std::vector<MNN::Session*> sessions;
    std::vector<MNN::Tensor *> inputs, outputs;
    std::vector<std::shared_ptr<MNN::Tensor>> givenTensors, expectedTensors, expectedTensorsSingle;
    std::set<MNN::SessionId> sessionIds;

    for (int i = 0; i < nets.size(); i++) {
        for (int j = 0; j < fuseCount; j++) {
            MNN::Session *session = nets[i]->createSession(config);
            sessions.push_back(session);
            inputs.push_back(nets[i]->getSessionInput(session, NULL));

            MNN::Tensor *input = inputs.back();
            auto inputShape = input->shape();
            if (inputShape[0] != batch) {
                inputShape[0] = batch;
                nets[i]->resizeTensor(input, inputShape);
            }
            sessionIds.insert(multiSession.addSession(session));
        }
    }

    multiSession.prepare();
    

    for (int i = 0; i < nets.size(); i++) {
        for (int j = 0; j < fuseCount; j++) {
            MNN::Session *session = sessions[i * fuseCount + j];
            outputs.push_back(nets[i]->getSessionOutput(session, NULL));

            givenTensors.push_back(std::shared_ptr<MNN::Tensor>(MNN::Tensor::createHostTensorFromDevice(inputs.back(), false)));
            setInputData(givenTensors.back().get(), 5.0f);
            expectedTensors.push_back(std::shared_ptr<MNN::Tensor>(MNN::Tensor::createHostTensorFromDevice(outputs.back(), false)));
            expectedTensorsSingle.push_back(std::shared_ptr<MNN::Tensor>(MNN::Tensor::createHostTensorFromDevice(outputs.back(), false)));

        }
    }


    for (int j = 0; j < inputs.size(); j++) {
        inputs[j]->copyFromHostTensor(givenTensors[j].get());
    }
    multiSession.runParallel(sessionIds);
    for (int j = 0; j < outputs.size(); j++) {
        outputs[j]->copyToHostTensor(expectedTensors[j].get());
    }

    for (int j = 0; j < inputs.size(); j++) {
        inputs[j]->copyFromHostTensor(givenTensors[j].get());
    }
    for (auto session : sessions) {
        session->run();
    }

    for (int j = 0; j < outputs.size(); j++) {
        outputs[j]->copyToHostTensor(expectedTensorsSingle[j].get());
    }

    
    for (int j = 0; j < outputs.size(); j++) {
        if(!MNN::TensorUtils::compareTensors(expectedTensors[j].get(), expectedTensorsSingle[j].get())) {
            std::cout << "Different tensor detected!" << std::endl;
            expectedTensors[j]->print();
            expectedTensorsSingle[j]->print();
        }
    }

    return {};
}

static std::vector<float> runOpCombinations(std::vector<std::shared_ptr<Interpreter>> nets, int loop, int warmup = 10, int forward = MNN_FORWARD_CPU, 
                           int numberThread = 4, int precision = 2, int combination = 2) {
    
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)precision;
    backendConfig.power = MNN::BackendConfig::Power_High;

    MNN::Backend::Info info;
    info.type = static_cast<MNNForwardType>(forward);
    info.numThread = numberThread;
    info.user = &backendConfig;

    auto backend = MNN::BackendFactory::create(info);

    MNN::ScheduleConfig config;
    config.backend = backend;

    std::vector<MNN::Session*> sessions;

    for (int i = 0; i < nets.size(); i++) {
        sessions.push_back(nets[i]->createSession(config));
    }

    std::map<Unit *, double> referenceTime;
    std::vector<std::pair<Unit*, Session*>> units;

    for (auto& session : sessions) {
        for (auto &pipeline : session->mPipelines) {
            for (auto & unit: pipeline->mUnits) {
                for (int i = 0; i < warmup; i++) {
                    unit->execute();
                }
                backend->onWaitFinish();
                double timeBegin = getTimeInUs();
                for (int i = 0; i < loop; i++) {
                    unit->execute();
                }
                backend->onWaitFinish();
                double timeEnd = getTimeInUs();
                referenceTime[unit.get()] = ((timeEnd - timeBegin)) / 1000.0 / (double)loop;

                // Only consider fusionable unit
                if (unit.get()->mExecution->fusionable()){
                    units.push_back({unit.get(), session});
                }
                //else {
                //    std::cout << "not fusable : " << unit->name() << std::endl;
                //}

                //std::cout << unit->name() << " : " << (timeEnd - timeBegin) / 1000.0 << std::endl;

                // TODO : Percentage of computation per model
            }
        }
    }

    std::vector<bool> v(units.size());
    std::fill(v.end() - combination, v.end(), true);

    do {
        std::set<MNN::Session *> targetSessionSet;
        std::set<MNN::Unit *> targetUnitSet;
        std::vector<std::vector<MNN::Unit *>> targetUnits;
        double sumReferenceTime = 0;
        // Select current combination of units
        for (int i = 0; i < v.size(); ++i) {
            if (v[i]) {
                targetUnits.push_back({units[i].first});
                targetUnitSet.insert(units[i].first);
                targetSessionSet.insert(units[i].second);
                sumReferenceTime += referenceTime[units[i].first];
            }
        }
        
        // Skips combination from same model
        if (targetUnitSet.size() != targetSessionSet.size()) {
            continue;
        }

        // Prepare multi-unit
        MNN::MultiUnit mu(targetUnits, backend.get());  

        mu.prepare();
        
        for (int i = 0; i < warmup; i++) {
            mu.execute();
        }
        backend->onWaitFinish();
        auto timeBegin = getTimeInUs();
        for (int i = 0; i < loop; i++) {
            mu.execute();
        }
        backend->onWaitFinish();

        auto timeEnd = getTimeInUs();

        for (auto& unit : targetUnitSet) {
            unit->mInputs[0]->printShape();
            unit->mOutputs[0]->printShape();
            std::cout << unit->name() + " ";
        }

        double fusedTime = ((timeEnd - timeBegin) / 1000.0) / loop;
        std::cout << " reference time : " << sumReferenceTime << " fused time : " << fusedTime << " efficiency : " << fusedTime / sumReferenceTime << std::endl;
    } while (std::next_permutation(v.begin(), v.end()));

    return {};
}

int main(int argc, const char* argv[]) {
   std::cout << "MNN benchmark" << std::endl;
    auto handle = dlopen("libMNN_CL.so", RTLD_NOW);
    int mode = 0;
    int loop = 10;
    int warmup = 10;
    MNNForwardType forward = MNN_FORWARD_CPU;
    int numberThread = 4;
    int precision = 2;
    int combination = 2;
    if (argc <= 2) {
        std::cout << "Usage: " << argv[0] << " models_folder [mode] [loop_count] [warmup] [forwardtype] [numberThread] [precision] [combination]" << std::endl;
        return 1;
    }
    if (argc >= 3) {
        mode = atoi(argv[2]);
    }
    if (argc >= 4) {
        loop = atoi(argv[3]);
    }
    if (argc >= 5) {
        warmup = atoi(argv[4]);
    }
    if (argc >= 6) {
        forward = static_cast<MNNForwardType>(atoi(argv[5]));
    }
    if (argc >= 7) {
        numberThread = atoi(argv[6]);
    }
    if (argc >= 8) {
        precision = atoi(argv[7]);
    }
    if (argc >= 9) {
        combination = atoi(argv[8]);  
    }
    std::cout << "Forward type: **" << forwardType(forward) << "** thread=" << numberThread << "** precision=" << precision << std::endl;

    std::cout << "--------> Benchmarking... loop = " << argv[3] << ", warmup = " << warmup << std::endl;

    std::vector<std::shared_ptr<MNN::Interpreter>> nets;
    if (mode == 0) {
        for (int i = 0; i < 2; i++) {
            auto x = _Input({1, 3, 224, 224}, NC4HW4);
            x = _Conv(1, 0, x, {3, 24}, {3, 3}, SAME, {2, 2}, {1, 1}, 1);
            x = _Conv(1, 0, x, {24, 16}, {3, 3}, SAME, {2, 2}, {1, 1}, 1);
            x = _Conv(1, 0, x, {16, 16}, {1, 1}, SAME, {2, 2}, {1, 1}, 1);
            x = _Conv(1, 0, x, {16, 1}, {1, 1}, SAME, {2, 2}, {1, 1}, 1);
            x = _Conv(1, 0, x, {1, 1}, {1, 1}, SAME, {2, 2}, {1, 1}, 1);
            x = _Conv(1, 0, x, {1, 16}, {1, 1}, SAME, {2, 2}, {1, 1}, 1);
            x = _MaxPool(x, {4, 4});
            x = _Convert(x, NC4HW4);
            x = _Convert(x, NC4HW4);
            nets.push_back(std::shared_ptr<MNN::Interpreter>(createFromVARP(x)));
        }
        runNet(nets, forward, numberThread, precision);
    } else if (mode == 1) {
        std::vector<Model> models = findModelFiles(argv[1]);
        for(auto& model : models) {
            nets.push_back(std::shared_ptr<MNN::Interpreter>(createFromModel(model)));
        }
        runOpCombinations(nets, loop, warmup, forward, numberThread, precision, combination);
    }


    return 0;
}
