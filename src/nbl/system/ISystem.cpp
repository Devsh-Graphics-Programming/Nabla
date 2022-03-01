#include "nbl/system/ISystem.h"
#include "nbl/system/CArchiveLoaderZip.h"
#include "nbl/system/CArchiveLoaderTar.h"


using namespace nbl;
using namespace nbl::system;


ISystem::ISystem(core::smart_refctd_ptr<ISystemCaller>&& caller) : m_dispatcher(this, std::move(caller))
{
    addArchiveLoader(core::make_smart_refctd_ptr<CArchiveLoaderZip>(core::smart_refctd_ptr<ISystem>(this), nullptr));
    addArchiveLoader(core::make_smart_refctd_ptr<CArchiveLoaderTar>(core::smart_refctd_ptr<ISystem>(this), nullptr));
}

core::smart_refctd_ptr<IFile> ISystem::loadBuiltinData(const std::string& builtinPath)
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

core::smart_refctd_ptr<IFile> ISystem::impl_loadEmbeddedBuiltinData(const std::string& builtinPath, const std::pair<const uint8_t*,size_t>& found)
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

void ISystem::addArchiveLoader(core::smart_refctd_ptr<IArchiveLoader>&& loader)
{
    const char** exts = loader->getAssociatedFileExtensions();
    uint32_t i = 0u;
    while (const char* e = exts[i++])
        m_loaders.perFileExt.insert(e, core::smart_refctd_ptr(loader));
    m_loaders.pushToVector(std::move(loader));
}

bool ISystem::exists(const system::path& filename, core::bitflag<IFile::E_CREATE_FLAGS> flags) const
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
    if (!writeUsage && std::get<IFileArchive*>(findFileInArchive(filename)))
        return true;
    // regular file
    return std::filesystem::exists(filename);
}

/*
    Returns true if the path is writable (e.g. if p is a path inside an archive the function will return true).
    The path existence is not checked.
*/
bool ISystem::isPathReadOnly(const system::path& p)
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


core::smart_refctd_ptr<IFile> ISystemCaller::createFile(core::smart_refctd_ptr<ISystem>&& sys, const std::filesystem::path& filename, core::bitflag<IFile::E_CREATE_FLAGS> flags)
{
    if (flags.value & IFile::ECF_READ)
    {        
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

std::pair<IFileArchive*,IFileArchive::SOpenFileParams> ISystem::findFileInArchive(const system::path& _path) const
{
    system::path path = std::filesystem::exists(_path) ? system::path(std::filesystem::canonical(_path.parent_path()).generic_string()) : _path.parent_path();

        
    while (!path.empty() && path.parent_path() != path) // going up the directory tree
    {
        system::path realPath = std::filesystem::exists(path) ? system::path(std::filesystem::canonical(path).generic_string()) : path;
        auto archives = m_cachedArchiveFiles.findRange(realPath);

        for (auto& archive : archives)
        {
            auto relative = std::filesystem::relative(_path, path);
            auto files = archive.second->getArchivedFiles();
            auto itemToFind = IFileArchive::SFileListEntry{ relative, relative, 0 };
            bool hasFile = std::binary_search(files.begin(), files.end(), itemToFind, [](const IFileArchive::SFileListEntry& l, const IFileArchive::SFileListEntry& r) { return l.fullName == r.fullName; });
            auto f = archive.second->asFile();
            if (f)
            {
                auto realPath = f->getFileName();
                auto absolute = (realPath / relative).generic_string();
                if (hasFile)
                {
                    auto f = archive.second;// ->readFile({ relative, absolute, "" });
                    return { f.get(),{relative,_path,""} };
                }
            }
            else
            {
                if (hasFile)
                {
                    auto f = archive.second;// ->readFile({ relative, _path, "" });
                    return { f.get(),{relative,_path,""} };
                }
            }
        }
            path = path.parent_path();
    }
    return {nullptr,{}};
}
