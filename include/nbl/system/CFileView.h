#ifndef C_MEMORY_FILE_H
#define C_MEMORY_FILE_H
#include <nbl/system/IFile.h>

namespace nbl::system
{
class CFileView : public IFile
{
public:
	CFileView(core::smart_refctd_ptr<ISystem>&& sys, const std::filesystem::path& _name, std::underlying_type_t<E_CREATE_FLAGS> _flags) : IFile(std::move(sys),_flags | ECF_COHERENT | ECF_MAPPABLE), m_name(_name)
	{
	}

	virtual const std::filesystem::path& getFileName() const override
	{
		return m_name;
	}

	void* getMappedPointer() override final { return m_buffer.data(); }
	const void* getMappedPointer() const override final { return m_buffer.data(); }

	size_t getSize() const override final
	{
		return m_buffer.size();
	}
protected:
	size_t read_impl(void* buffer, size_t offset, size_t sizeToRead) override final
	{
		if (offset + sizeToRead > m_buffer.size())
		{
			return 0u;
		}
		memcpy(buffer, m_buffer.data(), sizeToRead);
		return sizeToRead;
	}

	size_t write_impl(const void* buffer, size_t offset, size_t sizeToWrite) override final
	{
		if (offset + sizeToWrite > m_buffer.size())
		{
			m_buffer.resize(offset + sizeToWrite);
		}
		memcpy(m_buffer.data() + offset, buffer, sizeToWrite);
		return sizeToWrite;
	}
private:
	std::filesystem::path m_name;
	core::vector<uint8_t> m_buffer;
};

// TODO custom allocator memory file
}

#endif