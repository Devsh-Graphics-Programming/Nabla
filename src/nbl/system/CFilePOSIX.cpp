#include "nbl/system/CFilePOSIX.h"

using namespace nbl::system;

#ifdef __unix__ // WTF: can it be `defined(_NBL_PLATFORM_ANDROID_) | defined(_NBL_PLATFORM_LINUX_)` instead?
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>

CFilePOSIX::CFilePOSIX(
	core::smart_refctd_ptr<ISystem>&& sys,
	path&& _filename,
	const core::bitflag<E_CREATE_FLAGS> _flags,
	void* const _mappedPtr,
	const size_t _size,
	const native_file_handle_t _native
) : IFile(std::move(sys),std::move(_filename),_flags,_mappedPtr),
	m_size(_size), m_native(_native)
{
}

CFilePOSIX::~CFilePOSIX()
{
	if (m_mappedPtr)
		munmap(m_mappedPtr,m_size);
	close(m_native);
}

size_t CFilePOSIX::asyncRead(void* buffer, size_t offset, size_t sizeToRead)
{
	lseek(m_native, offset, SEEK_SET);
	return ::read(m_native, buffer, sizeToRead);
}

size_t CFilePOSIX::asyncWrite(const void* buffer, size_t offset, size_t sizeToWrite)
{
	lseek(m_native, offset, SEEK_SET);
	return ::write(m_native, buffer, sizeToWrite);
}
#endif