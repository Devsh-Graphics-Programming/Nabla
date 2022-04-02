// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_C_FILESYSTEM_INCLUDER_H_INCLUDED_
#define _NBL_ASSET_C_FILESYSTEM_INCLUDER_H_INCLUDED_

#include "nbl/asset/utils/IIncluder.h"

#include "nbl/system/IFile.h"

namespace nbl::asset
{

class CFilesystemIncluder : public IIncluder
{
    public:
        CFilesystemIncluder(system::ISystem* _sys) : m_system{_sys}
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
                m_system->createFile(future,_path.c_str(),system::IFile::ECF_READ);
                f = future.get();
                if (!f)
                    return {};
            }
            const size_t size = f->getSize();

            std::string contents(size,'\0');
            system::IFile::success_t succ;
            f->read(succ, contents.data(), 0, size);
            const bool success = bool(succ);
            assert(success);

            return contents;
        }

    private:
        system::ISystem* m_system;
};

}

#endif