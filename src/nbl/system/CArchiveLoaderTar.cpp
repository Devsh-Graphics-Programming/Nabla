#include "nbl/system/CArchiveLoaderTar.h"

namespace nbl::system
{
core::smart_refctd_ptr<IFile> CFileArchiveTar::readFile_impl(const SOpenFileParams& params)
{
    auto found = std::find_if(m_files.begin(), m_files.end(), [&params](const SFileListEntry& entry) { return params.filename == entry.fullName; });

    if(found != m_files.end())
    {
        uint8_t* buff = (uint8_t*)m_file->getMappedPointer() + found->offset;
        auto a = core::make_smart_refctd_ptr<CFileView<CNullAllocator>>(
            core::smart_refctd_ptr<ISystem>(m_system),
            params.absolutePath,
            IFile::ECF_READ_WRITE,
            buff,
            found->size);
        return a;
    }
    return nullptr;
}
uint32_t CFileArchiveTar::populateFileList()
{
    size_t readOffset;
    STarHeader fHead;
    m_files.clear();

    uint32_t pos = 0;
    while(pos + sizeof(STarHeader) < m_file->getSize())
    {
        // seek to next file header
        readOffset = pos;

        // read the header
        read_blocking(m_file.get(), &fHead, readOffset, sizeof fHead);

        // only add standard files for now
        if(fHead.Link == ETLI_REGULAR_FILE || ETLI_REGULAR_FILE_OLD)
        {
            std::string fullPath = "";
            fullPath.reserve(255);

            // USTAR archives have a filename prefix
            // may not be null terminated, copy carefully!
            if(!strncmp(fHead.Magic, "ustar", 5))
            {
                char* np = fHead.FileNamePrefix;
                while(*np && (np - fHead.FileNamePrefix) < 155)
                    fullPath += *np;
                np++;
            }

            // append the file name
            char* np = fHead.FileName;
            while(*np && (np - fHead.FileName) < 100)
            {
                fullPath += *np;
                np++;
            }

            // get size
            std::string sSize = "";
            sSize.reserve(12);
            np = fHead.Size;
            while(*np && (np - fHead.Size) < 12)
            {
                sSize += *np;
                np++;
            }

            uint32_t size = strtoul(sSize.c_str(), NULL, 8);
            if(errno == ERANGE)
                m_logger.log("File %s is too large", ILogger::ELL_WARNING, fullPath.c_str());

            // save start position
            uint32_t offset = pos + 512;

            // move to next file header block
            pos = offset + (size / 512) * 512 + ((size % 512) ? 512 : 0);

            // add file to list
            addItem(fullPath, offset, size, EAT_NULL);
        }
        else
        {
            // todo: ETLI_DIRECTORY, ETLI_LINK_TO_ARCHIVED_FILE

            // move to next block
            pos += 512;
        }
    }

    return m_files.size();
}
}