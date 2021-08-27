#include "nbl/system/ISystem.h"
#include "nbl/system/CArchiveLoaderZip.h"

namespace nbl::system
{
    ISystem::ISystem(core::smart_refctd_ptr<ISystemCaller>&& caller) : m_dispatcher(this, std::move(caller))
    {
        addArchiveLoader(core::make_smart_refctd_ptr<CArchiveLoaderZip>());
    }
    core::smart_refctd_ptr<IFile> ISystem::getFileFromArchive(const system::path& _path)
    {
        system::path path = _path.parent_path().string();

        
        bool isPathAlias = !std::filesystem::exists(path);
        while (!path.empty()) // going up the directory tree
        {
            system::path realPath = path;
            if (isPathAlias)
            {
                auto a = m_cachedPathAliases.findRange(path);
                if (a.empty())
                {
                    path = path.parent_path();
                    continue;
                }
                realPath = a.begin()->second;
            }
            auto archives = m_cachedArchiveFiles.findRange(realPath);
            for (auto& archive : archives)
            {
                auto relative = std::filesystem::relative(_path, path);
                auto files = archive.second->getArchivedFiles();
                auto requiredFile = std::find_if(files.begin(), files.end(), [&relative](const IFileArchive::SFileListEntry& entry) { return entry.fullName == relative; });
                if (requiredFile != files.end()) return archive.second->readFile({ relative, "" }); //TODO password
                auto nestedArchivePath = relative.parent_path();
                while (!nestedArchivePath.empty())
                {
                    auto arch = std::find_if(files.begin(), files.end(), [&nestedArchivePath](const IFileArchive::SFileListEntry& entry) { return entry.fullName == nestedArchivePath; });
                    if (arch != files.end())
                    {
                        // TODO: still not sure how to cope with nested archives
                        assert(false);
                    }
                    nestedArchivePath = nestedArchivePath.parent_path();
                }
            }
            path = path.parent_path();
        }
        return nullptr;
    }
}
