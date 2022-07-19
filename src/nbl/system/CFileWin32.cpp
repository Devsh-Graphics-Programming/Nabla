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
	const size_t _size,
	HANDLE _native,
	HANDLE _fileMappingObj
) : ISystemFile(std::move(sys),std::move(_filename),_flags,_mappedPtr),
	m_size(_size), m_native(_native), m_fileMappingObj(_fileMappingObj)
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