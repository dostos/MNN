#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <cfloat>
#include <map>
#include <cstring>
#include <cstdlib>
#include <sstream>
#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "MNN_generated.h"
#include <MNN/MNNForwardType.h>
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Expr.hpp>
#include "ExprModels.hpp"

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

static std::vector<float> runNet(VARP netOutput, const ScheduleConfig& config, int loop) {
    std::unique_ptr<NetT> netTable(new NetT);
    Variable::save({netOutput}, netTable.get());
    flatbuffers::FlatBufferBuilder builder(1024);
    auto offset = CreateNet(builder, netTable.get());
    builder.Finish(offset);
    const void* buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();
    std::unique_ptr<Interpreter> net(Interpreter::createFromBuffer(buf, size));
    auto session = net->createSession(config);
    net->releaseModel();
    auto inputTensor = net->getSessionInput(session, NULL);
    std::shared_ptr<Tensor> inputTensorHost(Tensor::createHostTensorFromDevice(inputTensor, false));
    int eleSize = inputTensorHost->elementSize();
    for (int i = 0; i < eleSize; ++i) {
        inputTensorHost->host<float>()[i] = i;
    }
    auto outputTensor = net->getSessionOutput(session, NULL);
    std::shared_ptr<Tensor> outputTensorHost(Tensor::createHostTensorFromDevice(outputTensor, false));

    std::vector<float> costs;
    
    inputTensorHost->print();

    // start run
    for (int i = 0; i < loop; ++i) {
        auto timeBegin = getTimeInUs();


        inputTensor->copyFromHostTensor(inputTensorHost.get());
        net->runSession(session);
        outputTensor->copyToHostTensor(outputTensorHost.get());

        auto timeEnd = getTimeInUs();
        costs.push_back((timeEnd - timeBegin) / 1000.0);
    }
    return costs;
}

int main(int argc, const char* argv[]) {
    std::cout << "MNN Expr Models benchmark" << std::endl;

    std::vector<MNNForwardType> forwards = {MNN_FORWARD_CPU, MNN_FORWARD_OPENCL};
    std::string forwardsString;

    if (argc > 1) {
        forwards.clear();

        forwardsString = argv[1];
        std::replace(forwardsString.begin(), forwardsString.end(), ',', ' ');
        std::stringstream ss(forwardsString);

        int forwardType = 0;
        while(ss >> forwardType) {
            forwards.push_back(static_cast<MNNForwardType>(forwardType));
        }
    }

    int loop = 1;
    if (argc > 2) {
        loop = atoi(argv[2]);
    }

    std::cout << "Forward type: ** "; 
    for(auto forward : forwards) {
        std::cout << forwardType(forward) << " , ";
    }
    std::cout << std::endl;

    for(auto forward : forwards) {
        ScheduleConfig config;
        config.type = forward;
        
        BackendConfig bnConfig;
        bnConfig.precision = BackendConfig::Precision_Low;
        bnConfig.power = BackendConfig::Power_High;
        config.backendConfig = &bnConfig;

        std::vector<float> costs = runNet(ConvExpr({1, 3, 7, 7}, 3, {3, 3}, {0, 0}, {1, 1}, {1, 1}), config, loop);
        
        displayStats(forwardType(forward), costs);
    }
    return 0;
}
