#ifndef ARGUMENTPARSER_H
#define ARGUMENTPARSER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <stdexcept>

// Parse AA=BB pairs from argc/argv
inline std::unordered_map<std::string, std::string> parseArgs(int argc, char* argv[]) {
    std::unordered_map<std::string, std::string> argMap;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        size_t pos = arg.find('=');
        if (pos != std::string::npos) {
            std::string key = arg.substr(0, pos);
            std::string value = arg.substr(pos + 1);
            argMap[key] = value;
        }
    }
    return argMap;
}

// Split a string by a given delimiter (default = ';')
inline std::vector<std::string> split(const std::string& s, char delimiter = ';') {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delimiter)) {
        if (!item.empty()) result.push_back(item);
    }
    return result;
}

// Convert string to type T (with specializations)
template <typename T>
inline T fromString(const std::string& s);

// Specializations for fromString:
template <>
inline int fromString<int>(const std::string& s) {
    return std::stoi(s);
}

template <>
inline double fromString<double>(const std::string& s) {
    return std::stod(s);
}

template <>
inline bool fromString<bool>(const std::string& s) {
    return (s == "1" || s == "true" || s == "yes");
}

template <>
inline std::string fromString<std::string>(const std::string& s) {
    return s;
}

// Get scalar value of type T or default
template <typename T>
inline T getOrDefault(const std::unordered_map<std::string, std::string>& args,
                     const std::string& key,
                     const T& defaultValue) {
    auto it = args.find(key);
    if (it != args.end()) {
        return fromString<T>(it->second);
    }
    return defaultValue;
}

template <typename T>
inline T getOrDefault(const std::unordered_map<std::string, std::string>& args,
                     const std::string& key) {
    auto it = args.find(key);
    if (it != args.end()) {
        return fromString<T>(it->second);
    }
    throw std::runtime_error("Missing required key: " + key);
}

template <>
inline unsigned int fromString<unsigned int>(const std::string& s) {
    return static_cast<unsigned int>(std::stoul(s));
}


// Get vector<T> from delimited string or default
template <typename T>
inline std::vector<T> getOrDefaultVector(const std::unordered_map<std::string, std::string>& args,
                                         const std::string& key,
                                         const std::vector<T>& defaultValue,
                                         char delimiter = ';') {
    auto it = args.find(key);
    if (it != args.end()) {
        std::vector<std::string> parts = split(it->second, delimiter);
        std::vector<T> result;
        for (const auto& part : parts) {
            result.push_back(fromString<T>(part));
        }
        return result;
    }
    return defaultValue;
}

template <typename T>
inline std::vector<T> getOrDefaultVector(const std::unordered_map<std::string, std::string>& args,
                                         const std::string& key,
                                         char delimiter = ';') {
    auto it = args.find(key);
    if (it != args.end()) {
        std::vector<std::string> parts = split(it->second, delimiter);
        std::vector<T> result;
        for (const auto& part : parts) {
            result.push_back(fromString<T>(part));
        }
        return result;
    }
    throw std::runtime_error("Missing required key: " + key);
}

#endif // ARGUMENTPARSER_H
