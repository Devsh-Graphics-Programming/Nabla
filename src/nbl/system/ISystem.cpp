#include "nbl/system/ISystem.h"
#include "nbl/system/CArchiveLoaderZip.h"
#include "nbl/system/CArchiveLoaderTar.h"

namespace nbl::system
{
    core::smart_refctd_ptr<IFile> ISystemCaller::createFile(core::smart_refctd_ptr<ISystem>&& sys, const std::filesystem::path& filename, core::bitflag<IFile::E_CREATE_FLAGS> flags)
    {
        if (flags.value & IFile::ECF_READ)
        {
            auto f = sys->openFileOpt_impl(filename, flags);
            if (f.get() != nullptr) return f;
            
            auto a = sys->getFileFromArchive(filename);
            if (a.get() != nullptr) return a;
        }
        system::path realname = filename;
        if (std::filesystem::exists(filename))
        {
            realname = std::filesystem::absolute(filename).generic_string();
        }
        return createFile_impl(std::move(sys), realname, flags);
    }
    ISystem::ISystem(core::smart_refctd_ptr<ISystemCaller>&& caller) : m_dispatcher(this, std::move(caller))
    {
        addArchiveLoader(core::make_smart_refctd_ptr<CArchiveLoaderZip>(core::smart_refctd_ptr<ISystem>(this), nullptr));
        addArchiveLoader(core::make_smart_refctd_ptr<CArchiveLoaderTar>(core::smart_refctd_ptr<ISystem>(this), nullptr));
    }
    core::smart_refctd_ptr<IFile> ISystem::getFileFromArchive(const system::path& _path)
    {
        system::path path = std::filesystem::exists(_path) ? system::path(std::filesystem::canonical(_path.parent_path()).generic_string()) : _path.parent_path();

        
        while (!path.empty() && path.parent_path() != path) // going up the directory tree
        {
            system::path realPath = std::filesystem::exists(path) ? system::path(std::filesystem::canonical(path).generic_string()) : path;
            auto archives = m_cachedArchiveFiles.findRange(realPath);

            for (auto& archive : archives)
            {
                realPath = archive.second->asFile()->getFileName();
                auto relative = std::filesystem::relative(_path, path);
                auto absolute = (realPath / relative).generic_string();
                auto files = archive.second->getArchivedFiles();
                // TODO: file list should be sorted by the path and you should be using a binary search !!!!!!
                auto requiredFile = std::find_if(files.begin(), files.end(), [&relative](const IFileArchive::SFileListEntry& entry) { return entry.fullName == relative; });
                if (requiredFile != files.end())
                {
                    auto f =  archive.second->readFile({ relative, absolute, "" });
                    if (f.get()) return f;
                }
            }
            path = path.parent_path();
        }
        return nullptr;
    }
}
