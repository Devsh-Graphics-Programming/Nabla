#ifndef	_NBL_SYSTEM_CFILEWIN32_H_INCLUDED_
#define	_NBL_SYSTEM_CFILEWIN32_H_INCLUDED_

#include "IFile.h"

#include "nbl/system/DefaultFuncPtrLoader.h"
#ifdef _NBL_PLATFORM_WINDOWS_

namespace nbl::system
{

class CFileWin32 : public IFile
{
		using base_t = IFile;
		using native_file_handle_t = HANDLE;
	private:
		DWORD m_allocGranularity;
		bool m_openedProperly = true;
		size_t m_size = 0;
		native_file_handle_t m_native = nullptr;
		HANDLE m_fileMappingObj = nullptr;
		mutable core::vector<void*> m_openedFileViews;
	public:
		CFileWin32(core::smart_refctd_ptr<ISystem>&& sys, const std::filesystem::path& _filename, core::bitflag<E_CREATE_FLAGS> _flags);
		~CFileWin32();
		// Inherited via IFile
		virtual size_t getSize() const override;
		virtual void* getMappedPointer() override;
		virtual const void* getMappedPointer() const override;
		bool isOpenedProperly() const { return m_openedProperly; }
	private:
		virtual size_t read_impl(void* buffer, size_t offset, size_t sizeToRead) override;
		virtual size_t write_impl(const void* buffer, size_t offset, size_t sizeToWrite) override;

		void seek(size_t bytesFromBeginningOfFile);
};

}
#endif

#endif