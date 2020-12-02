#include "KernelCompiler.hpp"
#include <regex>
#include <iostream>
#include <sstream>

namespace MNN {
namespace OpenCL {
KernelContent::KernelContent(std::string name, std::string source)
    :name(name), source(source) {}

KernelCompiler::KernelCompiler(std::string commonSource) {
    std::regex defineRegex(R"(#define (\w*)[(\s])");
    std::smatch regexMatch;
    std::sregex_iterator end;

    std::for_each(std::sregex_iterator(commonSource.begin(), commonSource.end(), defineRegex), std::sregex_iterator(), [&](const std::smatch &match) {
        mDefines.insert(match[1]);
    });
}

KernelCompiler::~KernelCompiler() {
    for (auto cache : mContentCaches) {
        delete cache.second;
    }
}

const KernelContent* KernelCompiler::parse(std::string kernelName, std::string source) {
    if (mContentCaches.find(kernelName) != mContentCaches.end()) {
        return mContentCaches[kernelName];
    } else {
        KernelContent* content = new KernelContent(kernelName, source);

        // Find kernel name (emptyspace)kernelName(emptyspace)(
        std::regex kernelNameRegex("\\s+" + kernelName + "\\s*\\(");
        std::smatch regexMatch;
        std::regex_search(source, regexMatch, kernelNameRegex);
        
        size_t argsStartOffset = regexMatch.position() + regexMatch.length();

        std::regex argsEndRegex("\\)\\s*\\{");
        std::string argsCandidate = source.substr(argsStartOffset);
        std::regex_search(argsCandidate, regexMatch, argsEndRegex);
        size_t argsEndOffset = regexMatch.position();
        content->args = source.substr(argsStartOffset, argsEndOffset);

        // Parse args 
        {  
            // Add , for convinience to match last arg
            std::string argsString = content->args + ",";
            std::regex argsRegex( R"((\w*)\s*,)");
            std::sregex_iterator end;

            std::for_each(std::sregex_iterator(argsString.begin(), argsString.end(), argsRegex), std::sregex_iterator(), [&](const std::smatch &match) {
                content->pureArgs.insert(match[1]);
            });
        }

        size_t contentStartOffset = source.find("{", argsEndOffset);
        size_t contentEndOffset = contentStartOffset + 1;

        int bracketCount = 1;

        while (bracketCount > 0) {
            contentEndOffset = source.find_first_of("{}", contentEndOffset);
            bracketCount += source[contentEndOffset++] == '{' ? 1 : -1;
        };

        content->content = source.substr(contentStartOffset - 1, contentEndOffset - contentStartOffset + 2);
        mContentCaches[kernelName] = content;
        return content;
    }
}

const KernelContent*  KernelCompiler::fuse(std::vector<const KernelContent* > kernels) {
    std::string fusedName;

    // TODO : Sort names & return index to reuse
    for (int i = 0; i < kernels.size(); i++) {
        fusedName += kernels[i]->name + (i == kernels.size() - 1 ? "" : "_");
    }

    std::string prefix = "__kernel void " + fusedName + "(";
    std::string fusedArgs;
    std::string fusedContents;
    std::set<std::string> fusedPureArgs;

    for (int i = 0; i < kernels.size(); i++) {
        std::string argsString = kernels[i]->args;

        fusedPureArgs.insert(kernels[i]->pureArgs.begin(), kernels[i]->pureArgs.end());

        std::string defineBatchIndex = "(" + std::to_string(i) + ")";
        // Replace batch indexes in define
        argsString = std::regex_replace(argsString, std::regex{"\\((\\d+)\\)"}, defineBatchIndex.c_str());

        std::string content = kernels[i]->content;

        // Add additional index to args
        for (auto arg : kernels[i]->pureArgs) {
            std::smatch match;
            std::regex_search(argsString, match, std::regex{"\\W(" + arg + ")\\W*"});
            argsString.replace(match[1].first, match[1].second, arg + std::to_string(i));
        }

        // TODO : Support 3D
        argsString = "__private const int2 offset" + std::to_string(i) + ",\n" + argsString;

        fusedArgs += argsString + (i == kernels.size() - 1 ? "" : ",\n");

        // Update arg names in a source
        for (auto arg : kernels[i]->pureArgs) {
            std::smatch match;
            while (std::regex_search(content, match, std::regex{"[^\\d\\w](" + arg + ")[^\\d\\w]"})) {
                content.replace(match[1].first, match[1].second, arg + std::to_string(i));
            };
        }

        // TODO : Support 3D
        // Add gws indexing checker
        content = std::string(i == 0 ? "if" : "else if") + " GLOBAL_ID_CONDITION_2_DIMS(" + std::to_string(i) + ")" + content;

        fusedContents += content;
    }

    std::string source = prefix + fusedArgs + ")\n {\n" + fusedContents + "}";

    KernelContent *fusedKernel = new KernelContent(fusedName,source);
    fusedKernel->args = fusedArgs;
    fusedKernel->content = fusedContents;
    fusedKernel->pureArgs = fusedPureArgs;

    mContentCaches[fusedName] = fusedKernel;
    return fusedKernel;
}   
}
}