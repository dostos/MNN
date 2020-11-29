#include "KernelParser.hpp"
#include <regex>
#include <iostream>
#include <sstream>

namespace MNN {
namespace OpenCL {
KernelParser::KernelParser(std::string commonSource) {
    std::regex defineRegex(R"(#define (\w*)[(\s])");
    std::smatch regexMatch;
    std::sregex_iterator end;

    std::for_each(std::sregex_iterator(commonSource.begin(), commonSource.end(), defineRegex), std::sregex_iterator(), [&](const std::smatch &match) {
        mDefines.insert(match[1]);
    });
}

KernelContent KernelParser::parse(std::string kernelName, std::string source) {
    KernelContent content;

    // Find kernel name (emptyspace)kernelName(emptyspace)(
    std::regex kernelNameRegex("\\s+" + kernelName + "\\s*\\(");
    std::smatch regexMatch;
    std::regex_search(source, regexMatch, kernelNameRegex);

    size_t argsStartOffset = regexMatch.position() + regexMatch.length();
    size_t argsEndOffset = source.find(")", argsStartOffset);

    content.args = source.substr(argsStartOffset, argsEndOffset - argsStartOffset);

    // Parse pure args (arguments - define)
    {   
        // Add , for convinience to match last arg
        std::string argsString = content.args + ",";
        std::regex argsRegex( R"((\w*)\s*,)");
        std::smatch regexMatch;
        std::sregex_iterator end;

        std::for_each(std::sregex_iterator(argsString.begin(), argsString.end(), argsRegex), std::sregex_iterator(), [&](const std::smatch &match) {
            content.pureArgs.insert(match[1]);
        });
    }

    size_t contentStartOffset = source.find("{", argsEndOffset);
    size_t contentEndOffset = contentStartOffset + 1;

    int bracketCount = 1;

    while (bracketCount > 0) {
        contentEndOffset = source.find_first_of("{}", contentEndOffset);
        bracketCount += source[contentEndOffset++] == '{' ? 1 : -1;
    };

    content.content = source.substr(contentStartOffset - 1, contentEndOffset - contentStartOffset + 1);

    return content;
}
}
}