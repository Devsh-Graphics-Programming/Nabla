#ifdef __unix__
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "nbl/system/CFilePOSIX.h"

namespace nbl::system
{
CFilePOSIX::CFilePOSIX(core::smart_refctd_ptr<ISystem>&& sys, const std::filesystem::path& _filename, core::bitflag<E_CREATE_FLAGS> _flags)
    : base_t(std::move(sys), _filename, _flags)
{
    const char* name_c_str = _filename.string().c_str();
    int createFlags = O_LARGEFILE;
    int mappingFlags;
    if((_flags.value & ECF_READ_WRITE) == ECF_READ_WRITE)
    {
        createFlags |= O_CREAT;
        createFlags |= O_RDWR;
        mappingFlags = PROT_WRITE | PROT_READ;
    }
    else if(_flags.value & ECF_WRITE)
    {
        createFlags |= O_CREAT;
        createFlags |= O_WRONLY;
        mappingFlags = PROT_WRITE;
    }
    else if(_flags.value & ECF_READ)
    {
        createFlags |= O_RDONLY;
        mappingFlags = PROT_READ;
    }
    if((m_flags & ECF_READ).value == ECF_READ)
    {
        if(std::filesystem::exists(_filename))
        {
            m_native = open(name_c_str, createFlags);
        }
        else
            m_openedProperly = false;
    }
    m_openedProperly = m_native >= 0;
    if(m_openedProperly)
    {
        struct stat sb;
        if(stat(name_c_str, &sb) == -1)
        {
            m_openedProperly = false;
        }
        else
        {
            m_size = sb.st_size;
        }
    }

    if(_flags.value & ECF_MAPPABLE && m_openedProperly)
    {
        m_memoryMappedObj = mmap((caddr_t)0, m_size, mappingFlags, MAP_PRIVATE, m_native, 0);
        if(m_memoryMappedObj == MAP_FAILED)
            m_openedProperly = false;
    }
}

CFilePOSIX::~CFilePOSIX()
{
    if(m_memoryMappedObj)
    {
        munmap(m_memoryMappedObj, m_size);
    }
    if(m_native != -1)
    {
        close(m_native);
    }
}

size_t CFilePOSIX::getSize() const
{
    return m_size;
}

void* CFilePOSIX::getMappedPointer()
{
    return m_memoryMappedObj;
}

const void* CFilePOSIX::getMappedPointer() const
{
    return m_memoryMappedObj;
}

size_t CFilePOSIX::read_impl(void* buffer, size_t offset, size_t sizeToRead)
{
    if(m_flags.value & ECF_MAPPABLE)
    {
        memcpy(buffer, (std::byte*)m_memoryMappedObj + offset, sizeToRead);
    }
    else
    {
        lseek(m_native, offset, SEEK_SET);
        ::read(m_native, buffer, sizeToRead);
    }
    return sizeToRead;
}

size_t CFilePOSIX::write_impl(const void* buffer, size_t offset, size_t sizeToWrite)
{
    if(m_flags.value & ECF_MAPPABLE)
    {
        memcpy((std::byte*)buffer + offset, m_memoryMappedObj, sizeToWrite);
    }
    else
    {
        lseek(m_native, offset, SEEK_SET);
        ::write(m_native, buffer, sizeToWrite);
    }
    return sizeToWrite;
}

}

#endif