#ifdef _NBL_PLATFORM_WINDOWS_

#include "nbl/system/CFileWin32.h"

#define LODWORD(_qw)    ((DWORD)(_qw))
#define HIDWORD(_qw)    ((DWORD)(((_qw) >> 32) & 0xffffffff))

nbl::system::CFileWin32::CFileWin32(core::smart_refctd_ptr<ISystem>&& sys, const std::filesystem::path& _filename, std::underlying_type_t<E_CREATE_FLAGS> _flags) : base_t(std::move(sys), _flags), m_filename{ _filename }
{
	DWORD access = m_flags | ECF_READ_WRITE ? GENERIC_READ | GENERIC_WRITE :
		(m_flags | ECF_READ ? GENERIC_READ : (m_flags | ECF_WRITE ? GENERIC_WRITE : 0));
	const bool canOpenWhenOpened = false;
	SECURITY_ATTRIBUTES secAttribs{ sizeof(SECURITY_ATTRIBUTES), nullptr, FALSE };
	m_native = CreateFile(m_filename.string().data(), access, canOpenWhenOpened, &secAttribs, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (m_native != INVALID_HANDLE_VALUE) [[likely]] // let this idle here until c++20 :)
	{
		m_size = GetFileSize(m_native, nullptr);
	}
	else [[unlikely]]
	{
		m_openedProperly = false;
	}

}

nbl::system::CFileWin32::~CFileWin32()
{
	CloseHandle(m_native);
}

size_t nbl::system::CFileWin32::getSize() const
{
	return m_size;
}

const std::filesystem::path& nbl::system::CFileWin32::getFileName() const
{
	return m_filename;
}

void* nbl::system::CFileWin32::getMappedPointer()
{
	return nullptr;
}

const void* nbl::system::CFileWin32::getMappedPointer() const
{
	return nullptr;
}

size_t nbl::system::CFileWin32::read_impl(void* buffer, size_t offset, size_t sizeToRead)
{
	seek(offset);
	DWORD numOfBytesRead;
	ReadFile(m_native, buffer, sizeToRead, &numOfBytesRead, nullptr);
	return numOfBytesRead;
}

size_t nbl::system::CFileWin32::write_impl(const void* buffer, size_t offset, size_t sizeToWrite)
{
	seek(offset);
	DWORD numOfBytesWritten;
	WriteFile(m_native, buffer, sizeToWrite, &numOfBytesWritten, nullptr);
	return numOfBytesWritten;
}

void nbl::system::CFileWin32::seek(size_t position)
{
	LONG hiDword = HIDWORD(position);
	SetFilePointer(m_native, position, &hiDword, FILE_BEGIN);
}
#endif