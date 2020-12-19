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
#include "core/TensorUtils.hpp"
#include "revertMNNModel.hpp"
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>


using namespace MNN;
using namespace MNN::Express;

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

static std::vector<float> runNets(std::vector<VARP> models, int loop, int warmup = 10, int forward = MNN_FORWARD_CPU, 
                           int numberThread = 4, int precision = 2, int batch = 1, int fuseCount = 1) {
    std::vector<std::shared_ptr<MNN::Interpreter>> nets;

    for (int i = 0; i < models.size(); i++) {
        nets.push_back(std::shared_ptr<MNN::Interpreter>(createFromVARP(models[i])));
    }
    
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
    std::vector<std::shared_ptr<MNN::Tensor>> givenTensors, expectedTensors;
    std::set<MNN::SessionId> sessionIds;

    for (int i = 0; i < models.size(); i++) {
        for (int j = 0; j < fuseCount; j++) {
            MNN::Session *session = nets[i]->createSession(config);
            sessions.push_back(session);
            inputs.push_back(nets[i]->getSessionInput(session, NULL));

            MNN::Tensor *input = inputs.back();
            auto inputShape = input->shape();
            if (inputShape[0] != batch) {
                inputShape[0] = batch;
                nets[i]->resizeTensor(input, inputShape);
                std::cout << "Resized to " << batch << std::endl;
            }
            sessionIds.insert(multiSession.addSession(session));
        }
    }

    multiSession.prepare();
    

    for (int i = 0; i < models.size(); i++) {
        for (int j = 0; j < fuseCount; j++) {
            MNN::Session *session = sessions[i * fuseCount + j];
            outputs.push_back(nets[i]->getSessionOutput(session, NULL));

            givenTensors.push_back(std::shared_ptr<MNN::Tensor>(MNN::Tensor::createHostTensorFromDevice(inputs.back(), false)));
            setInputData(givenTensors.back().get(), 5.0f);
            expectedTensors.push_back(std::shared_ptr<MNN::Tensor>(MNN::Tensor::createHostTensorFromDevice(outputs.back(), false)));
        }
    }


    for (int i = 0; i < warmup; ++i) {
        for (int j = 0; j < inputs.size(); j++) {
            inputs[j]->copyFromHostTensor(givenTensors[j].get());
        }
        multiSession.runParallel(sessionIds);
        for (int j = 0; j < outputs.size(); j++) {
            outputs[j]->copyToHostTensor(expectedTensors[j].get());
        }
    }

    std::vector<float> costs;
    for (int round = 0; round < loop; round++) {
        auto timeBegin = getTimeInUs();

        for (int j = 0; j < inputs.size(); j++) {
            inputs[j]->copyFromHostTensor(givenTensors[j].get());
        }
        multiSession.runParallel(sessionIds);
        for (int j = 0; j < outputs.size(); j++) {
            outputs[j]->copyToHostTensor(expectedTensors[j].get());
        }

        auto timeEnd = getTimeInUs();
        costs.push_back((timeEnd - timeBegin) / 1000.0);
    }

    for (int i = 0; i < models.size(); i++) {
        for (int j = 0; j < fuseCount - 1; j++) {
            if (!MNN::TensorUtils::compareTensors(expectedTensors[i * fuseCount + j].get(), expectedTensors[i * fuseCount + j + 1].get())) {
                std::cout << "Different tensor detected!" << std::endl;
            }
        }
    }

    return costs;
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
    int batch = 1;
    int fuseCount = 1;
    if (argc <= 2) {
        std::cout << "Usage: " << argv[0] << " models_folder [mode] [loop_count] [warmup] [forwardtype] [numberThread] [precision] [batch] [fuseCount]" << std::endl;
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
        batch = atoi(argv[8]);  
    }
    if (argc >= 10) {
        fuseCount = atoi(argv[9]);  
    }
    std::cout << "Forward type: **" << forwardType(forward) << "** thread=" << numberThread << "** precision=" << precision << std::endl;

    std::cout << "--------> Benchmarking... loop = " << argv[3] << ", warmup = " << warmup << std::endl;

    std::vector<VARP> models;

    for (int i = 0; i < 2; i++) {
        auto x = _Input({1, 3, 224, 224}, NC4HW4);
        x = _Conv(rand() % 5, rand() % 5, x, {3, 24}, {1, 1}, SAME, {2, 2}, {1, 1}, 1);
        models.push_back(x);
    }

    runNets(models, loop, warmup, forward, numberThread, precision, batch, fuseCount);

    return 0;
}
