#ifndef	CFILE_WIN32_H
#define	CFILE_WIN32_H
#include "IFile.h"
namespace nbl::system
{
class CFileWin32 : public IFile
{
	using base_t = IFile;
	using native_file_handle_t = HANDLE;
private:
	bool m_openedProperly = true;
	std::filesystem::path m_filename;
	size_t m_size = 0;
	native_file_handle_t m_native = nullptr;
public:
	CFileWin32(const std::filesystem::path& _filename, std::underlying_type_t<E_CREATE_FLAGS> _flags);
	~CFileWin32();
	static core::smart_refctd_ptr<CFileWin32> create(const std::string_view& _filename, std::underlying_type_t<E_CREATE_FLAGS> _flags)
	{
		auto createdFile = core::make_smart_refctd_ptr<CFileWin32>(_filename, _flags);
		if (createdFile->m_openedProperly) return createdFile;
		return nullptr;
	}
	// Inherited via IFile
	virtual size_t getSize() const override;
	virtual const std::filesystem::path& getFileName() const override;
	virtual void* getMappedPointer() override;
	virtual const void* getMappedPointer() const override;
private:
	virtual int32_t read(void* buffer, size_t offset, size_t sizeToRead) override;
	virtual int32_t write(const void* buffer, size_t offset, size_t sizeToWrite) override;

	void seek(size_t bytesFromBeginningOfFile);
};
}
#endif