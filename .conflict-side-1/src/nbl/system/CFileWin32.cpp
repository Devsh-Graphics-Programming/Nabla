#include "nbl/system/CFileWin32.h"

#ifdef _NBL_PLATFORM_WINDOWS_

#ifndef LODWORD
#	define LODWORD(_qw)    ((DWORD)(_qw))
#endif
#ifndef HIDWORD
#	define HIDWORD(_qw)    ((DWORD)(((_qw) >> 32u) & 0xffffffffu))
#endif

using namespace nbl::system;

CFileWin32::CFileWin32(
	core::smart_refctd_ptr<ISystem>&& sys,
	path&& _filename,
	const core::bitflag<E_CREATE_FLAGS> _flags,
	void* const _mappedPtr,
	HANDLE _native,
	HANDLE _fileMappingObj
) : ISystemFile(std::move(sys),std::move(_filename),_flags,_mappedPtr),
	m_native(_native), m_fileMappingObj(_fileMappingObj)
{
}

CFileWin32::~CFileWin32()
{
	if (m_mappedPtr)
		UnmapViewOfFile(m_mappedPtr);
	if (m_fileMappingObj)
		CloseHandle(m_fileMappingObj);
	CloseHandle(m_native);
}

inline auto CFileWin32::getLastWriteTime() const -> time_point_t
{
	FILETIME modified;
	if (!GetFileTime(m_native,nullptr,nullptr,&modified))
		return time_point_t::max();
	ULARGE_INTEGER ull;
	ull.LowPart = modified.dwLowDateTime;
	ull.HighPart = modified.dwHighDateTime;

	using namespace std::chrono;
	auto const duration = time_point<file_clock>::duration{ ull.QuadPart };
	const_cast<CFileWin32*>(this)->setLastWriteTime(clock_cast<time_point_t::clock>(time_point<file_clock>{ duration }));
	return m_modified.load();
}

inline size_t CFileWin32::getSize() const
{
	unsigned long hi;
	static_assert(sizeof(unsigned long)==4);
	size_t lo = GetFileSize(m_native,&hi);
	return (size_t(hi)<<32ull)|lo;
}

size_t CFileWin32::asyncRead(void* buffer, size_t offset, size_t sizeToRead)
{
	seek(offset);
	DWORD numOfBytesRead;
	ReadFile(m_native, buffer, sizeToRead, &numOfBytesRead, nullptr);
	return numOfBytesRead;
}
size_t CFileWin32::asyncWrite(const void* buffer, size_t offset, size_t sizeToWrite)
{
	seek(offset);
	DWORD numOfBytesWritten;
	WriteFile(m_native, buffer, sizeToWrite, &numOfBytesWritten, nullptr);
	return numOfBytesWritten;
}


void CFileWin32::seek(size_t position)
{
	LONG hiDword = HIDWORD(position);
	SetFilePointer(m_native,position,&hiDword,FILE_BEGIN);
}
#endif