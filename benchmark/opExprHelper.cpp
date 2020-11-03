#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <cfloat>
#include <map>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <dlfcn.h>
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
#include <source/core/Macro.h>
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

static void fillTensorNC4HW4(Tensor* t) {
    const int batch = t->batch();
    const int channel = t->channel();
    const int channel4 = UP_DIV(channel, 4);
    const int height = t->height();
    const int width = t->width();

    int count = 0;

    for (int n = 0; n < batch; n++) {
        int n_offset = n * channel4 * height * width;
        for (int c4 = 0; c4 < channel4; c4++) {
            int c4_offset = c4 * height * width;
            for (int y = 0; y < height; y++) {
                int y_offset = y * width;
                for (int x = 0; x < width; x++) {
                    int remainingChannel = channel - c4 * 4;
                    for (int c = 0; c < remainingChannel; c++) {
                        t->host<float>()[
                            (n_offset + c4_offset + y_offset + x) * 4 + c
                        ] = count++;
                    }
                }
            }
        }
    }

    MNN_ASSERT(count == batch * channel * height * width);
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

    memset(inputTensorHost->host<float>(), 0, eleSize);

    fillTensorNC4HW4(inputTensorHost.get());

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

    outputTensorHost->print();

    return costs;
}

int main(int argc, const char* argv[]) {
    auto handle = dlopen("libMNN_CL.so", RTLD_NOW);
    std::cout << "MNN Expr Models benchmark" << std::endl;

    std::vector<MNNForwardType> forwards = {MNN_FORWARD_CPU, MNN_FORWARD_OPENCL, MNN_FORWARD_VULKAN};
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

    std::vector<int> args = {
        1, // Loop
        7, // Input size
        3, // Input channel
        1, // Output channel
        3, // Kernel size
        1, // Stride
        1, // Dilate
        0 // Pad
    };

    for (int i = 2; i < argc; i++) {
        args[i - 2] = atoi(argv[i]);
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

        std::vector<float> costs = runNet(ConvExpr({1, args[2], args[1], args[1]}, args[3], {args[4], args[4]}, {args[5], args[5]}, {args[6], args[6]}, {args[7], args[7]}), config, args[0]);
        
        displayStats(forwardType(forward), costs);
    }
    return 0;
}
