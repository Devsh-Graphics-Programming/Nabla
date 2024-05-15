/*
# Copyright(c) 2024 DevSH Graphics Programming Sp.z O.O.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissionsand
# limitations under the License.
*/

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <argparse/argparse.hpp>
#include <nbl/core/xxHash256.h>

constexpr std::string_view NBL_FILE_ARG = "--file";

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("Compute & returns xxHash256 to stdout as json");

    program.add_argument(NBL_FILE_ARG.data())
        .required()
        .help("Input file path to hash for");

    try 
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) 
    {
        std::cerr << err.what() << std::endl << program;
        return 1;
    }
   
    std::filesystem::path filePath = program.get<std::string>(NBL_FILE_ARG.data());

    if (!std::filesystem::exists(filePath)) 
    {
        std::cerr << "File does not exist: " << filePath << std::endl;
        return 1;
    }

    const auto fileSize = std::filesystem::file_size(filePath);

    std::ifstream file(filePath, std::ios::binary);
    if (!file) 
    {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return 1;
    }

    std::vector<char> buffer(fileSize);
    if (!file.read(buffer.data(), fileSize)) {
        std::cerr << "Failed to read file: " << filePath << std::endl;
        return 1;
    }

    std::array<uint64_t, 4> hash = {};
    nbl::core::XXHash_256(buffer.data(), fileSize, hash.data());

    printf("{\"u64hash\": [%s,%s,%s,%s]}", std::to_string(hash[0]).c_str(), std::to_string(hash[1]).c_str(), std::to_string(hash[2]).c_str(), std::to_string(hash[3]).c_str());

    return 0;
}