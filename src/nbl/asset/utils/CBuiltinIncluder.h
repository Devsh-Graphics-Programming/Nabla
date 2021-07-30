// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_C_BUILTIN_INCLUDER_H_INCLUDED__
#define __NBL_ASSET_C_BUILTIN_INCLUDER_H_INCLUDED__

#include "CObjectCache.h"

#include "nbl/asset/utils/IIncluder.h"
#include "nbl/asset/utils/IGLSLEmbeddedIncludeLoader.h"

namespace nbl
{
namespace asset
{

class CBuiltinIncluder : public IIncluder
{
        core::smart_refctd_ptr<IGLSLEmbeddedIncludeLoader> m_default;

        using LoadersContainer = core::CMultiObjectCache<std::string, IBuiltinIncludeLoader*>;
        LoadersContainer m_loaders;

        static void loaderGrab(IBuiltinIncludeLoader* _ldr) { _ldr->grab(); }
        static void loaderDrop(IBuiltinIncludeLoader* _ldr) { _ldr->drop(); }

    public:
        CBuiltinIncluder(system::ISystem* s) : m_default(core::make_smart_refctd_ptr<IGLSLEmbeddedIncludeLoader>(s)), m_loaders(&loaderGrab, &loaderDrop)
        {
            m_searchDirectories.emplace_back("/");
        }

        //! No-op, cannot add search dirs to includer of builtins
        void addSearchDirectory(const system::path& _searchDir) override {}

        void addBuiltinLoader(core::smart_refctd_ptr<IBuiltinIncludeLoader>&& _loader)
        {
            using namespace std::string_literals;
            if (!_loader)
                return;

            m_loaders.insert(std::string(IIncludeHandler::BUILTIN_PREFIX) + _loader->getVirtualDirectoryName(), _loader.get());
        }

    protected:
        std::string getInclude_internal(const system::path& _path) const override
        {
            if (!IIncludeHandler::isBuiltinPath(_path))
                return {};

            const std::string relativePath = std::filesystem::relative(_path, system::path(IIncludeHandler::BUILTIN_PREFIX)).string();
            std::string path = _path.parent_path().string();
            std::string res;
            while (path != IIncludeHandler::BUILTIN_PREFIX) // going up the directory tree
            {
                auto capableLoadersRng = m_loaders.findRange(path);
                for (auto& loader : capableLoadersRng)
                {
                    if (!(res = loader.second->getBuiltinInclude(relativePath)).empty())
                        return res;
                }
                if (path.size()==0ull)
                    break;
                path.back() = 'x'; // get rid of trailing slash...
                path.erase(path.find_last_of('/')+1, std::string::npos); // ...and find the one before
            }
            return m_default->getBuiltinInclude(relativePath);
        }
};

}
}

#endif
