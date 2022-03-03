#ifndef _NBL_SYSTEM_I_FILE_H_INCLUDED_
#define _NBL_SYSTEM_I_FILE_H_INCLUDED_

#include "nbl/system/ISystem.h"

namespace nbl::system
{

class IFile : public IFileBase
{
	public:
		//
		virtual void read(ISystem::future_t<size_t>& fut, void* buffer, size_t offset, size_t sizeToRead) = 0;
		virtual void write(ISystem::future_t<size_t>& fut, const void* buffer, size_t offset, size_t sizeToWrite) = 0;

		//
		struct success_t
		{
			public:
				success_t() = default;
				~success_t() = default;
				inline explicit operator bool()
				{
					return m_internalFuture.get()==sizeToProcess;
				}
				inline bool operator!()
				{
					return m_internalFuture.get()!=sizeToProcess;
				}
			private:
				friend IFile;
				ISystem::future_t<size_t> m_internalFuture;
				size_t sizeToProcess;
		};
		void read(success_t& fut, void* buffer, size_t offset, size_t sizeToRead)
		{
			read(fut.m_internalFuture,buffer,offset,sizeToRead);
			fut.sizeToProcess = sizeToRead;
		}
		void write(success_t& fut, const void* buffer, size_t offset, size_t sizeToWrite)
		{
			write(fut.m_internalFuture,buffer,offset,sizeToWrite);
			fut.sizeToProcess = sizeToWrite;
		}

	protected:
		// this is an abstract interface class so this stays protected
		using IFileBase::IFileBase;
};

}

#endif
