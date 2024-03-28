#ifndef	_NBL_SYSTEM_CFILEWIN32_H_INCLUDED_
#define	_NBL_SYSTEM_CFILEWIN32_H_INCLUDED_

#include "nbl/system/ISystemFile.h"

namespace nbl::system
{

#ifdef _NBL_PLATFORM_WINDOWS_
class CFileWin32 : public ISystemFile
{
	public:
		CFileWin32(
			core::smart_refctd_ptr<ISystem>&& sys,
			path&& _filename,
			const core::bitflag<E_CREATE_FLAGS> _flags,
			void* const _mappedPtr,
			HANDLE _native,
			HANDLE _fileMappingObj
		);

		//
		time_point_t getLastWriteTime() const override;

		//
		size_t getSize() const override;

	protected:
		~CFileWin32();
		
		size_t asyncRead(void* buffer, size_t offset, size_t sizeToRead) override;
		size_t asyncWrite(const void* buffer, size_t offset, size_t sizeToWrite) override;

	private:
		void seek(const size_t bytesFromBeginningOfFile);

		HANDLE m_native;
		HANDLE m_fileMappingObj;
};
#endif

}

#endif