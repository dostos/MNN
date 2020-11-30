#ifndef KernelCompiler_hpp
#define KernelCompiler_hpp

#include <string>
#include <vector>
#include <set>
#include <map>

namespace MNN {
namespace OpenCL {
    struct KernelContent {
        KernelContent(std::string name, std::string source);
        std::string name;
        std::string source;
        std::set<std::string> pureArgs;
        std::string args;
        std::string content;
    };

    class KernelCompiler {
        public:
            KernelCompiler(std::string commonSource);
            ~KernelCompiler();
            const KernelContent* parse(std::string kernelName, std::string source);
            const KernelContent* fuse(std::vector<const KernelContent* > kernels);

        public:
            std::map<std::string, KernelContent *> mContentCaches;
            std::set<std::string> mDefines;
    };
}
}

#endif // KernelCompiler_hpp