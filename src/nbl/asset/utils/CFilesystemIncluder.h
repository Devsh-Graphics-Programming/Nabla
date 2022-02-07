// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_INCLUDER_H_INCLUDED__
#define __NBL_ASSET_C_INCLUDER_H_INCLUDED__

#include "nbl/asset/utils/IIncluder.h"
#include "nbl/system/ISystem.h"

namespace nbl
{
namespace asset
{
class CFilesystemIncluder : public IIncluder
{
public:
    CFilesystemIncluder(system::ISystem* _sys)
        : m_system{_sys}
    {
    }

    void addSearchDirectory(const system::path& _searchDir) override
    {
        std::filesystem::path absPath = std::filesystem::absolute(_searchDir);
        IIncluder::addSearchDirectory(absPath.string());
    }

    std::string getInclude_internal(const system::path& _path) const override
    {
        core::smart_refctd_ptr<system::IFile> f;
        {
            system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
            bool valid = m_system->createFile(future, _path.c_str(), system::IFile::ECF_READ);
            if(valid)
                f = future.get();
            if(!f)
                return {};
        }
        size_t size = f->getSize();
        std::string contents(size, '\0');
        system::future<size_t> future;
        f->read(future, contents.data(), 0, size);
        future.get();

        return contents;
    }

private:
    system::ISystem* m_system;
};

}
}

#endif