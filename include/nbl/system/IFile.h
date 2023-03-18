#ifndef _NBL_SYSTEM_I_FILE_H_INCLUDED_
#define _NBL_SYSTEM_I_FILE_H_INCLUDED_

#include "nbl/system/ISystem.h"

namespace nbl::system
{

class IFile : public IFileBase, private ISystem::IFutureManipulator
{
	public:
		//
		inline void read(ISystem::future_t<size_t>& fut, void* buffer, size_t offset, size_t sizeToRead)
		{
			const IFileBase* constThis = this;
			const auto* ptr = reinterpret_cast<const std::byte*>(constThis->getMappedPointer());
			if (ptr || sizeToRead==0ull)
			{
				const size_t size = getSize();
				if (offset+sizeToRead>size)
					sizeToRead = size-offset;
				memcpy(buffer,ptr+offset,sizeToRead);
				fake_notify(fut,sizeToRead);
			}
			else
				unmappedRead(fut,buffer,offset,sizeToRead);
		}
		inline void write(ISystem::future_t<size_t>& fut, const void* buffer, size_t offset, size_t sizeToWrite)
		{
			auto* ptr = reinterpret_cast<std::byte*>(getMappedPointer());
			if (ptr || sizeToWrite==0ull)
			{
				// TODO: growable mappings
				if (offset+sizeToWrite>getSize())
					sizeToWrite = getSize()-offset;
				memcpy(ptr+offset,buffer,sizeToWrite);
				fake_notify(fut,sizeToWrite);
			}
			else
				unmappedWrite(fut,buffer,offset,sizeToWrite);
		}

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

				inline size_t getSizeToProcess() const {return sizeToProcess;}

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

		//
		virtual void unmappedRead(ISystem::future_t<size_t>& fut, void* buffer, size_t offset, size_t sizeToRead)
		{
			fake_notify(fut,0ull);
		}
		virtual void unmappedWrite(ISystem::future_t<size_t>& fut, const void* buffer, size_t offset, size_t sizeToWrite)
		{
			fake_notify(fut,0ull);
		}
};

}

#endif
