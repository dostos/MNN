#ifndef KernelParser_hpp
#define KernelParser_hpp

#include <string>
#include <vector>
#include <set>

namespace MNN {
namespace OpenCL {
    struct KernelContent {
        std::set<std::string> pureArgs;
        std::string args;
        std::string content;
    };

    class KernelParser {
        public:
            KernelParser(std::string commonSource);
            KernelContent parse(std::string kernelName, std::string source);
        
        public:
            std::set<std::string> mDefines;
    };
}
}

#endif // KernelParser_hpp