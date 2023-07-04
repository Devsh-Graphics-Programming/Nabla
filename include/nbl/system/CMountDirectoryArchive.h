#ifndef _NBL_SYSTEM_C_MOUNT_DIRECTORY_ARCHIVE_H_INCLUDED_
#define _NBL_SYSTEM_C_MOUNT_DIRECTORY_ARCHIVE_H_INCLUDED_

#include "nbl/system/IFileArchive.h"

#include "nbl/system/IFile.h"

namespace nbl::system {



class CMountDirectoryArchive : public IFileArchive
{
    ISystem* m_system;

public:
    inline CMountDirectoryArchive(path&& _defaultAbsolutePath, system::logger_opt_smart_ptr&& logger, ISystem* system) :
        IFileArchive(std::move(_defaultAbsolutePath), std::move(logger))
    {
        m_system = system;
    }

    core::smart_refctd_ptr<IFile> getFile(const path& pathRelativeToArchive, const std::string_view& password) override
    {
        {
            //std::unique_lock(itemMutex); already inside `getItemFromPath`
            if (!getItemFromPath(pathRelativeToArchive))
                return nullptr;
        }
        system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
        m_system->createFile(future, m_defaultAbsolutePath / pathRelativeToArchive, system::IFile::ECF_READ);
        if (auto file = future.acquire())
            return *file;
    }

 
    SFileList listAssets() const override {
        auto items = m_system->listItemsInDirectory(m_defaultAbsolutePath);
        auto new_entries = std::make_shared<core::vector<SFileList::SEntry>>();
        for (auto item : items)
        {
            if (item.has_extension())
            {
                auto relpath = item.lexically_relative(m_defaultAbsolutePath);
                auto entry = SFileList::SEntry{ relpath, 0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu, EAT_NONE };
                new_entries->push_back(entry);
            }
        }
        setItemList(new_entries);

        return IFileArchive::listAssets();
    }

    void populateItemList(const path& p) const {
       
    }
};

} //namespace nbl::system
#endif