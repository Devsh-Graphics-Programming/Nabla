// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_INCLUDER_H_INCLUDED__
#define __NBL_ASSET_I_INCLUDER_H_INCLUDED__

#include <string>

#include "nbl/core/core.h"
#include "nbl/system/system.h"
#include "IFileSystem.h"

namespace nbl
{
namespace asset
{
class IIncluder : public core::IReferenceCounted
{
protected:
    core::vector<std::string> m_searchDirectories;

    virtual ~IIncluder() = default;

public:
    IIncluder()
        : m_searchDirectories{""} {}

    virtual void addSearchDirectory(const std::string& _searchDir) { m_searchDirectories.push_back(_searchDir); }

    std::string getIncludeStandard(const std::string& _path) const
    {
        for(const std::string& searchDir : m_searchDirectories)
        {
            io::path path = searchDir.c_str();
            path += _path.c_str();
            path = io::IFileSystem::flattenFilename(path);
            std::string res = getInclude_internal(path.c_str());
            if(!res.empty())
                return res;
        }
        return {};
    }
    std::string getIncludeRelative(const std::string& _path, const std::string& _workingDir) const
    {
        io::path path = _workingDir.c_str();
        if(!_workingDir.empty() && _workingDir.back() != '/')
            path += "/";
        path += _path.c_str();
        path = io::IFileSystem::flattenFilename(path);
        return getInclude_internal(path.c_str());
    }

protected:
    //! Always gets absolute path
    virtual std::string getInclude_internal(const std::string& _path) const = 0;
};

}
}

#endif