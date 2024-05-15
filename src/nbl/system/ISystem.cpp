#include "nbl/system/ISystem.h"
#include "nbl/system/ISystemFile.h"
#include "nbl/system/CFileView.h"
#ifdef NBL_EMBED_BUILTIN_RESOURCES
#include "nbl/builtin/CArchive.h"
#include "spirv/builtin/CArchive.h"
#include "boost/builtin/CArchive.h"
#endif // NBL_EMBED_BUILTIN_RESOURCES

#include "nbl/system/CArchiveLoaderZip.h"
#include "nbl/system/CArchiveLoaderTar.h"
#include "nbl/system/CMountDirectoryArchive.h"

using namespace nbl;
using namespace nbl::system;

ISystem::ISystem(core::smart_refctd_ptr<ISystem::ICaller>&& caller) : m_dispatcher(std::move(caller))
{
    addArchiveLoader(core::make_smart_refctd_ptr<CArchiveLoaderZip>(nullptr));
    addArchiveLoader(core::make_smart_refctd_ptr<CArchiveLoaderTar>(nullptr));
    
    #ifdef NBL_EMBED_BUILTIN_RESOURCES
    mount(core::make_smart_refctd_ptr<nbl::builtin::CArchive>(nullptr));
    mount(core::make_smart_refctd_ptr<spirv::builtin::CArchive>(nullptr));
    mount(core::make_smart_refctd_ptr<boost::builtin::CArchive>(nullptr));
    #else
    // TODO: absolute default entry paths? we should do something with it
    mount(core::make_smart_refctd_ptr<nbl::system::CMountDirectoryArchive>(NBL_BUILTIN_RESOURCES_DIRECTORY_PATH, nullptr, this), "nbl/builtin");
    mount(core::make_smart_refctd_ptr<nbl::system::CMountDirectoryArchive>(SPIRV_BUILTIN_RESOURCES_DIRECTORY_PATH, nullptr, this), "spirv");
    mount(core::make_smart_refctd_ptr<nbl::system::CMountDirectoryArchive>(BOOST_BUILTIN_RESOURCES_DIRECTORY_PATH, nullptr, this), "boost");
    #endif
}

bool ISystem::exists(const system::path& filename, const core::bitflag<IFile::E_CREATE_FLAGS> flags) const
{
    const bool writeUsage = flags.value&IFile::ECF_WRITE;
    
    // filename too long
    if (filename.string().size() >= sizeof(SRequestParams_CREATE_FILE::filename))
        return false;
    // archive file
    if (!writeUsage && findFileInArchive(filename).archive)
        return true;
    // regular file
    return std::filesystem::exists(filename);
}

bool ISystem::isPathReadOnly(const system::path& p) const
{
    // check all parent subpaths
    auto curPath = p;
    while (!curPath.empty() && curPath.parent_path() != curPath)
    {
        // then check for presence in an archive
        auto archives = m_cachedArchiveFiles.findRange(curPath);
        if (!archives.empty())
            return true;

        curPath = curPath.parent_path().generic_string();
    }

    return false;
}

core::vector<system::path> ISystem::listItemsInDirectory(const system::path& p) const
{
    core::vector<system::path> res;
    res.reserve(512u);

    auto addArchiveItems = [this,&res](const path& archPath, const path& dirPath) -> void
    {
        const auto archives = m_cachedArchiveFiles.findRange(archPath);
        for (auto& arch : archives)
        {
            const auto assets = static_cast<IFileArchive::SFileList::range_t>(arch.second->listAssets(std::filesystem::relative(dirPath,archPath)));
            for (auto& item : assets)
                res.push_back(archPath/item.pathRelativeToArchive);
        }
    };

    std::error_code err;
    const auto directories = std::filesystem::recursive_directory_iterator(p,err);
    if (!err)
    for (auto entry : directories)
    {
        res.push_back(entry.path());
        // entry could have been an archive
        addArchiveItems(entry.path(),p);
    }
    else
    {
        // check for part of subpath being an archive
        auto path = std::filesystem::exists(p) ? std::filesystem::canonical(p):p;
        // going up the directory tree
        while (!path.empty() && path.parent_path()!=path)
        {
            addArchiveItems(path,p);
            const bool ex = std::filesystem::exists(path);
            path = path.parent_path();
            if (ex)
                path = std::filesystem::canonical(path);
        }
    }
    return res;
}

bool ISystem::createDirectory(const system::path& p)
{
    return std::filesystem::create_directories(p);
}

bool ISystem::deleteDirectory(const system::path& p)
{
    if (std::filesystem::exists(p))
        return std::filesystem::remove_all(p);
    else
        return false;
}

std::error_code ISystem::moveFileOrDirectory(const system::path& oldPath, const system::path& newPath)
{
    std::error_code ec;
    std::filesystem::rename(oldPath,newPath,ec);
    return ec;
}

bool ISystem::copy(const system::path& from, const system::path& to)
{
    if (isPathReadOnly(to))
        return false;

    auto copyFile = [this](const system::path& from, const system::path& to) -> bool
    {
        future_t<core::smart_refctd_ptr<IFile>> readFileFut, writeFileFut;
        createFile(writeFileFut,to,IFile::ECF_WRITE);
        if (auto writeF=writeFileFut.acquire())
        {
            createFile(readFileFut,from,core::bitflag(IFile::ECF_READ)|IFile::ECF_COHERENT);
            if (auto readF=readFileFut.acquire())
            {
                // the consts here are super important
                const core::smart_refctd_ptr<const IFile>& readFptr = *readF;
                if (auto pSrc=readFptr->getMappedPointer())
                {
                    IFile::success_t bytesWritten;
                    (*writeF)->write(bytesWritten,pSrc,0,readFptr->getSize());
                    return bool(bytesWritten);
                }
            }
        }
        return false;
    };
    if (isPathReadOnly(from))
    {
        if (isDirectory(from))
        {
            const auto allItems = listItemsInDirectory(from);
            for (const auto& item : allItems)
            {
                auto relative = std::filesystem::relative(item,from);
                system::path targetName = (to/relative).generic_string();
                std::filesystem::create_directories(targetName.parent_path());
                if (!isDirectory(targetName))
                {
                    if (!copyFile(item,targetName))
                        return false;
                }
            }
            return true;
        }
        else
            return copyFile(from,to);
    }
    else
    {
		const auto copyOptions = std::filesystem::copy_options::recursive|std::filesystem::copy_options::overwrite_existing;
        std::error_code error;
        std::filesystem::copy(from, to, copyOptions, error);
        return static_cast<bool>(error);
    }
}

void ISystem::createFile(future_t<core::smart_refctd_ptr<IFile>>& future, std::filesystem::path filename, const core::bitflag<IFileBase::E_CREATE_FLAGS> flags, const std::string_view& accessToken)
{
    // canonicalize
    if (std::filesystem::exists(filename))
        filename = std::filesystem::canonical(filename);

    // try archives (readonly, for now)
    if (!(flags.value&IFile::ECF_WRITE))
    {
        const auto found = findFileInArchive(filename);
        if (found.archive)
        {
            auto file = found.archive->getFile(found.pathRelativeToArchive,flags,accessToken);
            if (file)
            {
                future.set_result(std::move(file));
                return;
            }
        }
    }

    //
    if (std::filesystem::exists(filename))
        filename = std::filesystem::absolute(filename).generic_string();
    if (filename.string().size()>=MAX_FILENAME_LENGTH)
    {
        future.set_result(nullptr);
        return;
    }


    SRequestParams_CREATE_FILE params;
    strcpy(params.filename,filename.string().c_str());
    params.flags = flags.value;
    m_dispatcher.request(&future,params);
}

core::smart_refctd_ptr<IFileArchive> ISystem::openFileArchive(core::smart_refctd_ptr<IFile>&& file, const std::string_view& password)
{
    // the file backing the archive needs to be readable
    if ((file->getFlags()&IFile::ECF_READ) == 0)
        return nullptr;

    // try the archive loaders by using the extension first
    auto ext = system::extension_wo_dot(file->getFileName());
    auto loaders = m_loaders.perFileExt.findRange(ext);
    for (auto& loader : loaders)
    {
        auto arch = loader.second->createArchive(std::move(file),password);
        if (arch)
            return arch;
    }
    // try again, without regard for extension
    for (auto& loader : m_loaders.vector)
    {
        auto arch = loader->createArchive(std::move(file),password);
        if (arch)
            return arch;
    }
    return nullptr;
}

ISystem::FoundArchiveFile ISystem::findFileInArchive(const system::path& absolutePath) const
{
    system::path path = std::filesystem::exists(absolutePath) ? std::filesystem::canonical(absolutePath.parent_path()):absolutePath.parent_path();
    // going up the directory tree
    while (!path.empty() && path.parent_path()!=path)
    {
        path = std::filesystem::exists(path) ? std::filesystem::canonical(path):path;

        const auto archives = m_cachedArchiveFiles.findRange(path);
        for (auto& archive : archives)
        {
            const auto relative = std::filesystem::relative(absolutePath,path);
            const auto items = static_cast<IFileArchive::SFileList::range_t>(archive.second->listAssets());

            const IFileArchive::SFileList::SEntry itemToFind = { relative };
            auto found = std::lower_bound(items.begin(), items.end(), itemToFind);
            if (found!=items.end() && found->pathRelativeToArchive==relative)
                return {archive.second.get(),relative};
        }
        path = path.parent_path();
    }
    return { nullptr,{} };
}


void ISystem::CAsyncQueue::process_request(base_t::future_base_t* _future_base, SRequestType& req)
{
    std::visit([=](auto& visitor) {
        using retval_t = std::remove_reference_t<decltype(visitor)>::retval_t;
        visitor(base_t::future_storage_cast<retval_t>(_future_base),m_caller.get());
    }, req.params);
}
void ISystem::SRequestParams_CREATE_FILE::operator()(core::StorageTrivializer<retval_t>* retval, ICaller* _caller)
{
    retval->construct(_caller->createFile(filename,flags));
}
void ISystem::SRequestParams_READ::operator()(core::StorageTrivializer<retval_t>* retval, ICaller* _caller)
{
    retval->construct(file->asyncRead(buffer,offset,size));
}
void ISystem::SRequestParams_WRITE::operator()(core::StorageTrivializer<retval_t>* retval, ICaller* _caller)
{
    retval->construct(file->asyncWrite(buffer,offset,size));
}

bool ISystem::ICaller::invalidateMapping(IFile* file, size_t offset, size_t size)
{
    const auto flags = file->getFlags();
    if (!file || !(flags&IFile::ECF_MAPPABLE))
        return false;
    else if (flags&IFile::ECF_COHERENT)
        return true;
    return invalidateMapping_impl(file,offset,size);
}
bool ISystem::ICaller::flushMapping(IFile* file, size_t offset, size_t size)
{
    const auto flags = file->getFlags();
    if (!file || !(flags&IFile::ECF_MAPPABLE))
        return false;
    else if (flags&IFile::ECF_COHERENT)
        return true;

    const bool retval = flushMapping_impl(file,offset,size);
    file->setLastWriteTime();
    return retval;
}

void  ISystem::unmountBuiltins() {

    auto removeByKey = [&, this](const char* s) {
        auto range = m_cachedArchiveFiles.findRange(s);
        std::vector<core::smart_refctd_ptr<IFileArchive>> items_to_remove = {}; //is it always just 1 item?
        for (auto it = range.begin(); it != range.end(); ++it)
        {
            items_to_remove.push_back(it->second);
        }
        for (size_t i = 0; i < items_to_remove.size(); i++)
        {
            m_cachedArchiveFiles.removeObject(items_to_remove[i], s);
        }
    };
    removeByKey("nbl/builtin");
    removeByKey("spirv");
    removeByKey("boost");
    
}