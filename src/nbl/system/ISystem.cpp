#include "nbl/system/ISystem.h"
#include "nbl/system/ISystemFile.h"
#include "nbl/system/CFileView.h"

//#include "nbl/system/CArchiveLoaderZip.h"
//#include "nbl/system/CArchiveLoaderTar.h"


using namespace nbl;
using namespace nbl::system;


ISystem::ISystem(core::smart_refctd_ptr<ISystem::ICaller>&& caller) : m_dispatcher(std::move(caller))
{
    //addArchiveLoader(core::make_smart_refctd_ptr<CArchiveLoaderZip>(nullptr));
    //addArchiveLoader(core::make_smart_refctd_ptr<CArchiveLoaderTar>(nullptr));
}

core::smart_refctd_ptr<IFile> ISystem::loadBuiltinData(const std::string& builtinPath) const
{
#ifdef _NBL_EMBED_BUILTIN_RESOURCES_
    return impl_loadEmbeddedBuiltinData(builtinPath,nbl::builtin::get_resource_runtime(builtinPath));
#else
    constexpr auto pathPrefix = "nbl/builtin/";
    auto pos = builtinPath.find(pathPrefix);
    std::string path;
    if (pos != std::string::npos)
        path = builtinResourceDirectory + builtinPath.substr(pos + strlen(pathPrefix));
    else
        path = builtinResourceDirectory + builtinPath;

    future_t<core::smart_refctd_ptr<IFile>> fut;
    createFile(future, path.c_str(), core::bitflag<IFile::E_CREATE_FLAGS>(IFile::ECF_READ) :: IFile::ECF_MAPPABLE);
    auto file = fut.get();
    if (file.get())
    { 
        return file;
    }
    return nullptr;
#endif
}

core::smart_refctd_ptr<IFile> ISystem::impl_loadEmbeddedBuiltinData(const std::string& builtinPath, const std::pair<const uint8_t*,size_t>& found) const
{
#ifdef _NBL_EMBED_BUILTIN_RESOURCES_
    if (found.first && found.second)
    {
        auto fileView = core::make_smart_refctd_ptr<CFileView<CNullAllocator>>(
            core::smart_refctd_ptr<ISystem>(this),
            builtinPath, 
            IFile::ECF_READ,
            found.first,
            found.second
        );
        return fileView;
    }
#endif
    return nullptr;
}

bool ISystem::exists(const system::path& filename, const core::bitflag<IFile::E_CREATE_FLAGS> flags) const
{
    const bool writeUsage = flags.value&IFile::ECF_WRITE;
#ifdef _NBL_EMBED_BUILTIN_RESOURCES_
    std::pair<const uint8_t*, size_t> found = nbl::builtin::get_resource_runtime(filename.string());
    if (!writeUsage && found.first && found.second)
        return true;
#endif
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
        // first check if its a builtin path
    #ifdef _NBL_EMBED_BUILTIN_RESOURCES_
        std::pair<const uint8_t*,size_t> found = nbl::builtin::get_resource_runtime(curPath.string());
        if (found.first && found.second)
            return true;
    #endif

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
    res.reserve(128u);

    // TODO: check for path being a builtin
    if (isPathReadOnly(p)) // TODO: better check for archives
    {
        auto curPath = p;
        while (!curPath.empty() && curPath.parent_path() != curPath)
        {
            auto archives = m_cachedArchiveFiles.findRange(curPath);
            for (auto& arch : archives)
            {
                auto rel = std::filesystem::relative(p, arch.first);
                auto res =  arch.second->listAssets(rel.generic_string().c_str());
                std::for_each(res.begin(), res.end(), [&arch](system::path& p) {p = arch.first / p; });
                return res;
            }

            curPath = curPath.parent_path().generic_string();
        }
    }
    else
    {
        uint32_t fileCount = std::distance(std::filesystem::recursive_directory_iterator(p), std::filesystem::recursive_directory_iterator{});
        for (auto entry : std::filesystem::recursive_directory_iterator(p))
            res.push_back(entry.path());
    }
    // TODO: recurse into any archives which could have been found!
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
        createFile(readFileFut,from,core::bitflag(IFile::ECF_READ)|IFile::ECF_COHERENT);
        createFile(writeFileFut,to,IFile::ECF_WRITE);
        auto readF = readFileFut.get();
        auto writeF = writeFileFut.get();
        if (!readF || !readF->getMappedPointer() || !writeF)
            return false;

        IFile::success_t bytesWritten;
        writeF->write(bytesWritten,readF->getMappedPointer(),0,readF->getSize());
        return bool(bytesWritten);
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
    // try builtins
    if (!(flags.value&IFile::ECF_WRITE))
    {
        auto file = loadBuiltinData(filename.string());
        if (file)
        {
            future.notify(std::move(file));
            return;
        }
    }
    // try archives
    if (flags.value&IFile::ECF_READ)
    {
        const auto found = findFileInArchive(filename);
        if (found.archive)
        {
            auto file = found.archive->getFile(found.pathRelativeToArchive,accessToken);
            if (file)
            {
                future.notify(std::move(file));
                return;
            }
        }
    }

    //
    if (std::filesystem::exists(filename))
        filename = std::filesystem::absolute(filename).generic_string();
    if (filename.string().size()>=SRequestParams_CREATE_FILE::MAX_FILENAME_LENGTH)
    {
        future.notify(nullptr);
        return;
    }

    SRequestParams_CREATE_FILE params;
    strcpy(params.filename,filename.string().c_str());
    params.flags = flags.value;
        
    m_dispatcher.request(future,params);
}

core::smart_refctd_ptr<IFileArchive> ISystem::openFileArchive(core::smart_refctd_ptr<IFile>&& file, const std::string_view& password)
{
    // the file backing the archive needs to be readable
    if (file->getFlags()&IFile::ECF_READ == 0)
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
            const auto items = archive.second->listAssets();

            const IFileArchive::SListEntry itemToFind = { relative };
            auto found = std::lower_bound(items.begin(), items.end(), itemToFind);
            if (found!=items.end() && found->pathRelativeToArchive==relative)
                return {archive.second.get(),relative};
        }
        path = path.parent_path();
    }
    return { nullptr,{} };
}


void ISystem::CAsyncQueue::process_request(SRequestType& req)
{
    switch (req.type)
    {
        case ERT_CREATE_FILE:
        {
            auto& p = std::get<SRequestParams_CREATE_FILE>(req.params);
            base_t::notify_future<core::smart_refctd_ptr<IFile>>(req,m_caller->createFile(p.filename,p.flags));
        }
        break;
        case ERT_READ:
        {
            auto& p = std::get<SRequestParams_READ>(req.params);
            base_t::notify_future<size_t>(req,p.file->asyncRead(p.buffer, p.offset, p.size));
        }
        break;
        case ERT_WRITE:
        {
            auto& p = std::get<SRequestParams_WRITE>(req.params);
            base_t::notify_future<size_t>(req,p.file->asyncWrite(p.buffer, p.offset, p.size));
        }
        break;
    }
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
    return flushMapping_impl(file,offset,size);
}