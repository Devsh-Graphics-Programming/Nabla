#ifndef C_MEMORY_FILE_H
#define C_MEMORY_FILE_H
#include <nbl/system/IFile.h>
//#include <nbl/system/IFileViewAllocator.h>

namespace nbl::system
{
template<typename allocator_t>
class CFileView : public IFile
{
protected:
	allocator_t allocator;
	size_t m_size;
public:
	CFileView(CFileView<allocator_t>&& other) : IFile(std::move(other.m_system), path(other.getFileName()), other.m_flags), m_size(other.m_size), m_buffer(other.m_buffer) 
	{
		other.m_buffer = nullptr;
	}
	CFileView(core::smart_refctd_ptr<ISystem>&& sys, path&& _name, core::bitflag<E_CREATE_FLAGS> _flags, size_t fileSize) : 
		IFile(std::move(sys), std::move(_name), _flags | ECF_COHERENT | ECF_MAPPABLE),
		m_size(fileSize)
	{
		m_buffer = (std::byte*)allocator.alloc(fileSize);
	}
	~CFileView()
	{
		allocator.dealloc(m_buffer);
	}

	void* getMappedPointer() override final { return m_buffer; }
	const void* getMappedPointer() const override final { return m_buffer; }

	size_t getSize() const override final
	{
		return m_size;
	}
protected:
	size_t read_impl(void* buffer, size_t offset, size_t sizeToRead) override final
	{
		if (offset + sizeToRead > m_size)
		{
			return 0u;
		}
		memcpy(buffer, m_buffer + offset, sizeToRead);
		return sizeToRead;
	}

	size_t write_impl(const void* buffer, size_t offset, size_t sizeToWrite) override final
	{
		if (offset + sizeToWrite > m_size)
		{
			return 0;
		}
		memcpy(m_buffer + offset, buffer, sizeToWrite);
		return sizeToWrite;
	}
protected:
	std::byte* m_buffer;
};

class CNullAllocator;
template<>
class CFileView<CNullAllocator> : public IFile
{
	size_t m_size;
public:
	CFileView(CFileView<CNullAllocator>&& other) : IFile(std::move(other.m_system), path(other.getFileName()), other.m_flags), m_size(other.m_size), m_buffer(other.m_buffer)
	{
		other.m_buffer = nullptr;
	}
	CFileView(core::smart_refctd_ptr<ISystem>&& sys, path&& _name, core::bitflag<E_CREATE_FLAGS> _flags, void* buffer, size_t fileSize) : IFile(std::move(sys), std::move(_name), _flags | ECF_COHERENT | ECF_MAPPABLE), m_size(fileSize), m_buffer((std::byte*)buffer)
	{
	}
	~CFileView() = default;

	void* getMappedPointer() override final { return m_buffer; }
	const void* getMappedPointer() const override final { return m_buffer; }

	size_t getSize() const override final
	{
		return m_size;
	}
protected:
	size_t read_impl(void* buffer, size_t offset, size_t sizeToRead) override final
	{
		if (offset + sizeToRead > m_size)
		{
			return 0u;
		}
		memcpy(buffer, m_buffer + offset, sizeToRead);
		return sizeToRead;
	}

	size_t write_impl(const void* buffer, size_t offset, size_t sizeToWrite) override final
	{
		if (offset + sizeToWrite > m_size)
		{
			return 0;
		}
		memcpy(m_buffer + offset, buffer, sizeToWrite);
		return sizeToWrite;
	}
private:
	std::byte* m_buffer;
};
}

#endif