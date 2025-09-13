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
				set_result(fut,sizeToRead);
			}
			else
				unmappedRead(fut,buffer,offset,sizeToRead);
		}
		//
		inline void write(ISystem::future_t<size_t>& fut, const void* buffer, size_t offset, size_t sizeToWrite)
		{
			auto* ptr = reinterpret_cast<std::byte*>(getMappedPointer());
			setLastWriteTime();
			if (ptr || sizeToWrite==0ull)
			{
				// TODO: growable mappings
				if (offset+sizeToWrite>getSize())
					sizeToWrite = getSize()-offset;
				memcpy(ptr+offset,buffer,sizeToWrite);
				set_result(fut,sizeToWrite);
			}
			else
				unmappedWrite(fut,buffer,offset,sizeToWrite);
			setLastWriteTime();
		}

		//! Less verbose future handling
		struct success_t
		{
			public:
				success_t() = default;
				~success_t() = default;

				inline size_t getBytesToProcess() const {return sizeToProcess;}

				inline size_t getBytesProcessed(const bool block=true) const
				{
					if (block && !m_internalFuture.wait())
						return 0ull;
					return *m_internalFuture.get();
				}

				inline explicit operator bool()
				{
					return getBytesProcessed()==getBytesToProcess();
				}
				inline bool operator!()
				{
					return getBytesProcessed()!=getBytesToProcess();
				}

			private:
				// cannot move in memory or pointers go boom
				success_t(const success_t&) = delete;
				success_t(success_t&&) = delete;
				success_t& operator=(const success_t&) = delete;
				success_t& operator=(success_t&&) = delete;

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
			set_result(fut,0ull);
		}
		virtual void unmappedWrite(ISystem::future_t<size_t>& fut, const void* buffer, size_t offset, size_t sizeToWrite)
		{
			set_result(fut,0ull);
		}
};

}

#endif
