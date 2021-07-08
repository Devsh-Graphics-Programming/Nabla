#ifndef C_MEMORY_FILE_H
#define C_MEMORY_FILE_H
#include <nbl/system/IFile.h>

namespace nbl::system
{
class CFileView : public IFile
{
public:
	CFileView(const std::filesystem::path& _name, std::underlying_type_t<E_CREATE_FLAGS> _flags) : IFile(nullptr,_flags | ECF_COHERENT | ECF_MAPPABLE), m_name(_name)
	{
		assert(false); // TODO: fix the filename of this source file
	}

	virtual const std::filesystem::path& getFileName() const override
	{
		return m_name;
	}

	void* getMappedPointer() override final { return m_buffer.data(); }
	const void* getMappedPointer() const override final { return m_buffer.data(); }

	int32_t read(void* buffer, size_t offset, size_t sizeToRead) override final
	{
		if (offset + sizeToRead > m_buffer.size())
		{
			return 0u;
		}
		memcpy(buffer, m_buffer.data(), sizeToRead);
		return sizeToRead;
	}

	int32_t write(const void* buffer, size_t offset, size_t sizeToWrite) override final
	{
		if (offset + sizeToWrite > m_buffer.size())
		{
			m_buffer.resize(offset + sizeToWrite);
		}
		memcpy(m_buffer.data() + offset, buffer, sizeToWrite);
		return sizeToWrite;
	}
	size_t getSize() const override final
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