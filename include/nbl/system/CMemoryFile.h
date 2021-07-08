#ifndef C_MEMORY_FILE_H
#define C_MEMORY_FILE_H
#include <nbl/system/IFile.h>

namespace nbl::system
{
class CFileView : public IFile
{
public:
	CFileView(const std::filesystem::path& _name, std::underlying_type_t<E_CREATE_FLAGS> _flags) : IFile(_flags | ECF_COHERENT | ECF_MAPPABLE), m_name(_name)
	{
	}

	virtual const std::filesystem::path& getFileName() const override
	{
		return m_name;
	}

	virtual void* getMappedPointer() override { return m_buffer.data(); }
	virtual const void* getMappedPointer() const override { return m_buffer.data(); }

	virtual int32_t read(void* buffer, size_t offset, size_t sizeToRead) override
	{
		if (offset + sizeToRead > m_buffer.size())
		{
			return 0u;
		}
		memcpy(buffer, m_buffer.data(), sizeToRead);
		return sizeToRead;
	}

	virtual int32_t write(const void* buffer, size_t offset, size_t sizeToWrite) override
	{
		if (offset + sizeToWrite > m_buffer.size())
		{
			m_buffer.resize(offset + sizeToWrite);
		}
		memcpy(m_buffer.data() + offset, buffer, sizeToWrite);
		return sizeToWrite;
	}
	virtual size_t getSize() const
	{
		return m_buffer.size();
	}
private:
	std::filesystem::path m_name;
	core::vector<uint8_t> m_buffer;
};

// TODO custom allocator memory file
}

#endif