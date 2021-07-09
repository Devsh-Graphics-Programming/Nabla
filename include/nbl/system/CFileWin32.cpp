#include "C:/dev/work/Nabla-vs/src/nbl/CMakeFiles/Nabla.dir/Debug/cmake_pch.hxx"
#include "CFileWin32.h"

#define LODWORD(_qw)    ((DWORD)(_qw))
#define HIDWORD(_qw)    ((DWORD)(((_qw) >> 32) & 0xffffffff))

nbl::system::CFileWin32::CFileWin32(const std::string_view& _filename, std::underlying_type_t<E_CREATE_FLAGS> _flags) : base_t(_flags), m_filename{ _filename }
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

int32_t nbl::system::CFileWin32::read(void* buffer, size_t offset, size_t sizeToRead)
{
	seek(offset);
	DWORD numOfBytesRead;
	ReadFile(m_native, buffer, sizeToRead, &numOfBytesRead, nullptr);
	return numOfBytesRead;
}

int32_t nbl::system::CFileWin32::write(const void* buffer, size_t offset, size_t sizeToWrite)
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
