#ifdef _NBL_PLATFORM_WINDOWS_

#include "nbl/system/CFileWin32.h"

#define LODWORD(_qw)    ((DWORD)(_qw))
#define HIDWORD(_qw)    ((DWORD)(((_qw) >> 32) & 0xffffffff))

nbl::system::CFileWin32::CFileWin32(core::smart_refctd_ptr<ISystem>&& sys, const std::filesystem::path& _filename, std::underlying_type_t<E_CREATE_FLAGS> _flags) : base_t(std::move(sys), _flags), m_filename{ _filename }
{
	SYSTEM_INFO info;
	GetSystemInfo(&info);
	m_allocGranularity = info.dwAllocationGranularity;

	DWORD access = m_flags | ECF_READ_WRITE ? FILE_GENERIC_READ | FILE_GENERIC_WRITE :
		(m_flags | ECF_READ ? FILE_GENERIC_READ : (m_flags | ECF_WRITE ? FILE_GENERIC_WRITE : 0));
	const bool canOpenWhenOpened = false;
	SECURITY_ATTRIBUTES secAttribs{ sizeof(SECURITY_ATTRIBUTES), nullptr, FALSE };
	
	system::path p = m_filename;
	if (p.is_absolute()) p.make_preferred(); // Replace "/" separators with "\"
	m_native = CreateFile(p.string().data(), access, canOpenWhenOpened, &secAttribs, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (m_native != INVALID_HANDLE_VALUE) [[likely]] // let this idle here until c++20 :)
	{
		m_size = GetFileSize(m_native, nullptr);
	}
	else [[unlikely]]
	{
		m_openedProperly = false;
	}
	if (m_flags & ECF_MAPPABLE)
	{
		DWORD access = (m_flags | ECF_READ_WRITE || m_flags | ECF_WRITE) ? PAGE_READWRITE :
			(m_flags | ECF_READ ? PAGE_READONLY : 0);
		m_openedProperly &= access != 0;
		/*
		TODO: should think of a better way to cope with the max size of a file mapping object (those two zeroes after `access`). 
		For now it equals the size of a file so it'll work fine for archive reading, but if we try to
		write outside those boungs, things will go bad.
		*/
		m_fileMappingObj = CreateFileMapping(m_native, nullptr, access, 0, 0, _filename.string().c_str()); 
		m_openedProperly &= m_fileMappingObj != nullptr;
	}
	

}

nbl::system::CFileWin32::~CFileWin32()
{
	for (void* view : m_openedFileViews)
	{
		UnmapViewOfFile(view);
	}
	CloseHandle(m_native);
	CloseHandle(m_fileMappingObj);
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
	void* view = MapViewOfFile(m_fileMappingObj, FILE_MAP_READ, 0, 0, m_size);
	m_openedFileViews.push_back(view);
	return view;
}

const void* nbl::system::CFileWin32::getMappedPointer() const
{
	void* view = MapViewOfFile(m_fileMappingObj, FILE_MAP_READ, 0, 0, m_size);
	m_openedFileViews.push_back(view);
	return view;
}

size_t nbl::system::CFileWin32::read_impl(void* buffer, size_t offset, size_t sizeToRead)
{
	if (m_flags & ECF_MAPPABLE)
	{
		auto viewOffset = (offset / m_allocGranularity) * m_allocGranularity;
		offset = offset % m_allocGranularity;
		DWORD l = LODWORD(viewOffset), h = HIDWORD(viewOffset);
		std::byte* fileView = (std::byte*)MapViewOfFile(m_fileMappingObj, FILE_MAP_READ, h, l, offset + sizeToRead);
		m_openedFileViews.push_back(fileView);
		if (fileView == nullptr)
		{
			assert(false);
			return 0;
		}
		std::copy<std::byte*>(fileView + offset, (std::byte*)fileView + offset + sizeToRead, (std::byte*)buffer);
		return sizeToRead;
	}
	else
	{
		seek(offset);
		DWORD numOfBytesRead;
		ReadFile(m_native, buffer, sizeToRead, &numOfBytesRead, nullptr);
		return numOfBytesRead;
	}
}

size_t nbl::system::CFileWin32::write_impl(const void* buffer, size_t offset, size_t sizeToWrite)
{
	if (m_flags & ECF_MAPPABLE)
	{
		auto viewOffset = (offset / m_allocGranularity) * m_allocGranularity;
		offset += offset % m_allocGranularity;
		std::byte* fileView = (std::byte*)MapViewOfFile(m_fileMappingObj, FILE_MAP_WRITE, HIWORD((DWORD)viewOffset), LOWORD((DWORD)viewOffset), sizeToWrite);
		std::copy((std::byte*)buffer, (std::byte*)buffer + sizeToWrite, fileView);
		return sizeToWrite;
	}
	else
	{
		seek(offset);
		DWORD numOfBytesWritten;
		WriteFile(m_native, buffer, sizeToWrite, &numOfBytesWritten, nullptr);
		return numOfBytesWritten;
	}
}

void nbl::system::CFileWin32::seek(size_t position)
{
	LONG hiDword = HIDWORD(position);
	SetFilePointer(m_native, position, &hiDword, FILE_BEGIN);
}
#endif