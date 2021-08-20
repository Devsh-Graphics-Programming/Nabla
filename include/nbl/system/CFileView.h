#ifndef C_MEMORY_FILE_H
#define C_MEMORY_FILE_H
#include <nbl/system/IFile.h>

namespace nbl::system
{
template<typename allocator_t>
class CFileView : public IFile
{
	static_assert(std::is_base_of_v<IFileViewAllocator, allocator_t>);
	allocator_t allocator;
	size_t m_size;
public:
	CFileView(core::smart_refctd_ptr<ISystem>&& sys, const std::filesystem::path& _name, std::underlying_type_t<E_CREATE_FLAGS> _flags, size_t fileSize) : IFile(std::move(sys),_flags | ECF_COHERENT | ECF_MAPPABLE), m_name(_name), m_size(fileSize)
	{
		m_buffer = (std::byte*)allocator.alloc(fileSize);
	}
	~CFileView()
	{
		allocator.dealloc(m_buffer);
	}
	virtual const std::filesystem::path& getFileName() const override
	{
		return m_name;
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
		memcpy(buffer, m_buffer, sizeToRead);
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
	std::filesystem::path m_name;
	std::byte* m_buffer;
};

// TODO custom allocator memory file
}

#endif