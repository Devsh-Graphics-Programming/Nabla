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
    
    core::vector<SListEntry> populateItemList(const path& p) {
        std::unique_lock lock(itemMutex);
        auto items = m_system->listItemsInDirectory(m_defaultAbsolutePath/p);
        m_items.clear();

        for (auto item : items)
        {
            if (item.has_extension())
            {
                auto p2 = item.lexically_relative(m_defaultAbsolutePath);
                auto entry = SListEntry{ p2,0xdeadbeefu,0xdeadbeefu,0xdeadbeefu ,EAT_NONE };
                m_items.push_back(entry);
            }
        }
      
    }
};

} //namespace nbl::system