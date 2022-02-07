// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_INCLUDER_H_INCLUDED__
#define __NBL_ASSET_C_INCLUDER_H_INCLUDED__

#include "nbl/asset/utils/IIncluder.h"
#include "IFileSystem.h"

namespace nbl
{
namespace asset
{
class CFilesystemIncluder : public IIncluder
{
public:
    CFilesystemIncluder(io::IFileSystem* _fs)
        : m_filesystem{_fs}
    {
    }

    void addSearchDirectory(const std::string& _searchDir) override
    {
        io::path absPath = m_filesystem->getAbsolutePath(_searchDir.c_str());
        IIncluder::addSearchDirectory(absPath.c_str());
    }

    std::string getInclude_internal(const std::string& _path) const override
    {
        auto f = m_filesystem->createAndOpenFile(_path.c_str());
        if(!f)
            return {};
        std::string contents(f->getSize(), '\0');
        f->read(&contents.front(), f->getSize());

        f->drop();

        return contents;
    }

private:
    io::IFileSystem* m_filesystem;
};

}
}

#endif