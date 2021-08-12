#ifndef	_NBL_SYSTEM_CFILEWIN32_H_INCLUDED_
#define	_NBL_SYSTEM_CFILEWIN32_H_INCLUDED_

#include "IFile.h"


#ifdef _NBL_PLATFORM_WINDOWS_
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
		CFileWin32(core::smart_refctd_ptr<ISystem>&& sys, const std::filesystem::path& _filename, std::underlying_type_t<E_CREATE_FLAGS> _flags);
		~CFileWin32();
		// Inherited via IFile
		virtual size_t getSize() const override;
		virtual const std::filesystem::path& getFileName() const override;
		virtual void* getMappedPointer() override;
		virtual const void* getMappedPointer() const override;
	private:
		virtual size_t read_impl(void* buffer, size_t offset, size_t sizeToRead) override;
		virtual size_t write_impl(const void* buffer, size_t offset, size_t sizeToWrite) override;

		void seek(size_t bytesFromBeginningOfFile);
};

}
#endif

#endif