#ifndef _NBL_SYSTEM_C_FILE_POSIX_H_INCLUDED_
#define _NBL_SYSTEM_C_FILE_POSIX_H_INCLUDED_
#if defined(_NBL_PLATFORM_ANDROID_) | defined(_NBL_PLATFORM_LINUX_)
#include "IFile.h"

namespace nbl::system
{
	class CFilePOSIX : public IFile
	{
		using base_t = IFile;
		using native_file_handle_t = int;
		using native_file_mapping_handle_t = void*;
	private:
		bool m_openedProperly = true;
		std::filesystem::path m_filename;
		size_t m_size = 0;
		native_file_handle_t m_native = -1;
		native_file_mapping_handle_t m_memoryMappedObj;
	public:
		CFilePOSIX(core::smart_refctd_ptr<ISystem>&& sys, const std::filesystem::path& _filename, core::bitflag<E_CREATE_FLAGS> _flags);
		~CFilePOSIX();
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
#endif // _NBL_SYSTEM_C_FILE_POSIX_H_INCLUDED
#endif
