// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_ASSET_I_GLSL_EMBEDDED_INCLUDE_LOADER_H_INCLUDED__
#define __NBL_ASSET_ASSET_I_GLSL_EMBEDDED_INCLUDE_LOADER_H_INCLUDED__

#include "nbl/system/system.h"
#include "IFileSystem.h"

#include "nbl/asset/utils/IBuiltinIncludeLoader.h"

namespace nbl
{
namespace asset
{
class IGLSLEmbeddedIncludeLoader : public IBuiltinIncludeLoader
{
protected:
    virtual ~IGLSLEmbeddedIncludeLoader() = default;

    inline core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
    {
        std::string pattern(getVirtualDirectoryName());
        pattern += ".*";
        HandleFunc_t tmp = [this](const std::string& _name) -> std::string { return getFromDiskOrEmbedding(_name); };
        return {{std::regex{pattern}, std::move(tmp)}};
    }

    static core::vector<std::string> parseArgumentsFromPath(const std::string& _path)
    {
        core::vector<std::string> args;

        std::stringstream ss{_path};
        std::string arg;
        while(std::getline(ss, arg, '/'))
            args.push_back(std::move(arg));

        return args;
    }

    io::IFileSystem* fs;

public:
    IGLSLEmbeddedIncludeLoader(io::IFileSystem* filesystem)
        : fs(filesystem) {}

    //
    const char* getVirtualDirectoryName() const override { return ""; }

    //
    inline std::string getFromDiskOrEmbedding(const std::string& _name) const
    {
        auto path = "nbl/builtin/" + _name;
        auto data = fs->loadBuiltinData(path);
        if(!data)
            return "";
        auto begin = reinterpret_cast<const char*>(data->getPointer());
        auto end = begin + data->getSize();
        return std::string(begin, end);
    }
};

}
}

#endif
