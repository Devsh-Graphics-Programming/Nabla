#ifndef _NBL_SYSTEM_C_FILE_POSIX_H_INCLUDED_
#define _NBL_SYSTEM_C_FILE_POSIX_H_INCLUDED_

#include "nbl/system/ISystemFile.h"

namespace nbl::system
{

#if defined(_NBL_PLATFORM_ANDROID_) | defined(_NBL_PLATFORM_LINUX_)
class CFilePOSIX : public ISystemFile
{
	public:
		using native_file_handle_t = int;
		CFilePOSIX(
			core::smart_refctd_ptr<ISystem>&& sys,
			path&& _filename,
			const core::bitflag<E_CREATE_FLAGS> _flags,
			void* const _mappedPtr,
			const size_t _size,
			const native_file_handle_t _native
		);

		// This is wrong! should re-query every time you call!
		inline size_t getSize() const override {return m_size;}

	protected:
		~CFilePOSIX();

		//
		size_t asyncRead(void* buffer, size_t offset, size_t sizeToRead) override;
		size_t asyncWrite(const void* buffer, size_t offset, size_t sizeToWrite) override;

	private:
		void seek(const size_t bytesFromBeginningOfFile);

		const size_t m_size; // this is wrong!
		const native_file_handle_t m_native;
};
#endif

}

#endif // _NBL_SYSTEM_C_FILE_POSIX_H_INCLUDED