#ifndef _NBL_SYSTEM_C_MOUNT_DIRECTORY_ARCHIVE_H_INCLUDED_
#define _NBL_SYSTEM_C_MOUNT_DIRECTORY_ARCHIVE_H_INCLUDED_

#include "nbl/system/IFileArchive.h"

#include "nbl/system/IFile.h"

namespace nbl::system
{


class CMountDirectoryArchive : public IFileArchive
{
        ISystem* m_system;

    public:
        inline CMountDirectoryArchive(path&& _defaultAbsolutePath, system::logger_opt_smart_ptr&& logger, ISystem* system) :
            IFileArchive(std::move(_defaultAbsolutePath), std::move(logger))
        {
            m_system = system;
        }

 
        SFileList listAssets() const override
        {
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

    protected:		
		inline core::smart_refctd_ptr<IFile> getFile_impl(const SFileList::found_t& found, const core::bitflag<IFile::E_CREATE_FLAGS> flags, const std::string_view& password) override
		{
            system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
            m_system->createFile(future,m_defaultAbsolutePath/found->pathRelativeToArchive,flags);
            if (auto file=future.acquire())
                return *file;

            return nullptr;
        }
};

} //namespace nbl::system
#endif