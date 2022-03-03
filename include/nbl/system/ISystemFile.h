#ifndef _NBL_SYSTEM_I_SYSTEM_FILE_H_INCLUDED_
#define _NBL_SYSTEM_I_SYSTEM_FILE_H_INCLUDED_


#include "nbl/system/IFile.h"


namespace nbl::system
{

class ISystemFile : public IFile, private ISystem::IFutureManipulator
{
	public:
		//
		inline void read(ISystem::future_t<size_t>& fut, void* buffer, size_t offset, size_t sizeToRead) final override
		{
			if (getFlags()&ECF_MAPPABLE)
			{
				const size_t size = getSize();
				if (offset+sizeToRead>size)
					sizeToRead = size-offset;
				memcpy(buffer,reinterpret_cast<const std::byte*>(m_mappedPtr)+offset,sizeToRead);
				fake_notify(fut,sizeToRead);
			}
			else
			{
				ISystem::SRequestParams_READ params;
				params.buffer = buffer;
				params.file = this;
				params.offset = offset;
				params.size = sizeToRead;
				m_system->m_dispatcher.request(fut,params);
			}
		}
		inline void write(ISystem::future_t<size_t>& fut, const void* buffer, size_t offset, size_t sizeToWrite) final override
		{
			if (getFlags()&ECF_MAPPABLE)
			{
				// TODO: growable mappings
				if (offset+sizeToWrite>getSize())
					sizeToWrite = getSize()-offset;
				memcpy(reinterpret_cast<std::byte*>(m_mappedPtr)+offset,buffer,sizeToWrite);
				fake_notify(fut,sizeToWrite);
			}
			else
			{
				ISystem::SRequestParams_WRITE params;
				params.buffer = buffer;
				params.file = this;
				params.offset = offset;
				params.size = sizeToWrite;
				m_system->m_dispatcher.request(fut,params);
			}
		}

	protected:
		// the ISystem is the factory, so this stays protected
		explicit ISystemFile(
			core::smart_refctd_ptr<ISystem>&& _system,
			path&& _filename,
			const core::bitflag<E_CREATE_FLAGS> _flags,
			void* const _mappedPtr
		) : IFile(std::move(_filename),_flags), m_system(std::move(_system)), m_mappedPtr(_mappedPtr) {}
		explicit ISystemFile(
			core::smart_refctd_ptr<ISystem>&& _system,
			const path& _filename,
			const core::bitflag<E_CREATE_FLAGS> _flags,
			void* const _mappedPtr
		) : IFile(std::move(_filename),_flags), m_system(_system), m_mappedPtr(_mappedPtr) {}
		
		//
		inline void* getMappedPointer_impl() override {return m_mappedPtr;}
		inline const void* getMappedPointer_impl() const override {return m_mappedPtr;}

		//
		//friend class ISystem::ICaller;
		virtual size_t asyncRead(void* buffer, size_t offset, size_t sizeToRead) = 0;
		virtual size_t asyncWrite(const void* buffer, size_t offset, size_t sizeToWrite) = 0;

		core::smart_refctd_ptr<ISystem> m_system;
		void* m_mappedPtr;
};

}

#endif
