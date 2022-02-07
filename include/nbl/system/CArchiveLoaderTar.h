#ifndef __NBL_SYSTEM_C_ARCHIVE_LOADER_TAR_H_INCLUDED__
#define __NBL_SYSTEM_C_ARCHIVE_LOADER_TAR_H_INCLUDED__
#include "IFileArchive.h"

namespace nbl::system
{
enum E_TAR_LINK_INDICATOR
{
    ETLI_REGULAR_FILE_OLD = 0,
    ETLI_REGULAR_FILE = '0',
    ETLI_LINK_TO_ARCHIVED_FILE = '1',  // Hard link
    ETLI_SYMBOLIC_LINK = '2',
    ETLI_CHAR_SPECIAL_DEVICE = '3',
    ETLI_BLOCK_SPECIAL_DEVICE = '4',
    ETLI_DIRECTORY = '5',
    ETLI_FIFO_SPECIAL_FILE = '6',
    ETLI_CONTIGUOUS_FILE = '7'
};

// byte-align structures
#include "nbl/nblpack.h"

struct STarHeader
{
    char FileName[100];
    char FileMode[8];
    char UserID[8];
    char GroupID[8];
    char Size[12];
    char ModifiedTime[12];
    char Checksum[8];
    char Link;
    char LinkName[100];
    char Magic[6];
    char USTARVersion[2];
    char UserName[32];
    char GroupName[32];
    char DeviceMajor[8];
    char DeviceMinor[8];
    char FileNamePrefix[155];
} PACK_STRUCT;

// Default alignment
#include "nbl/nblunpack.h"

class CFileArchiveTar : public IFileArchive
{
public:
    CFileArchiveTar(core::smart_refctd_ptr<system::IFile>&& file, core::smart_refctd_ptr<ISystem>&& system, system::logger_opt_smart_ptr&& logger = nullptr)
        : IFileArchive(std::move(file), std::move(system), std::move(logger))
    {
        if(m_file.get())
        {
            populateFileList();
        }
        setFlagsVectorSize(m_files.size());
    }
    virtual core::smart_refctd_ptr<IFile> readFile_impl(const SOpenFileParams& params) override;

private:
    uint32_t populateFileList();
};

class CArchiveLoaderTar : public IArchiveLoader
{
public:
    CArchiveLoaderTar(core::smart_refctd_ptr<ISystem>&& system, system::logger_opt_smart_ptr&& logger)
        : IArchiveLoader(std::move(system), std::move(logger)) {}
    virtual bool isALoadableFileFormat(IFile* file) const override
    {
        return file->getFileName().extension() == ".tar";
    }

    virtual const char** getAssociatedFileExtensions() const override
    {
        static const char* ext[]{"tar", nullptr};
        return ext;
    }

private:
    virtual core::smart_refctd_ptr<IFileArchive> createArchive_impl(core::smart_refctd_ptr<system::IFile>&& file, const std::string_view& password) const override
    {
        core::smart_refctd_ptr<IFileArchive> archive = nullptr;
        if(file.get())
        {
            archive = core::make_smart_refctd_ptr<CFileArchiveTar>(std::move(file), core::smart_refctd_ptr<ISystem>(m_system));
        }
        return archive;
    }
};
}

#endif