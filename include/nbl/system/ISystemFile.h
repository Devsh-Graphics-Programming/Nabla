#ifndef _NBL_SYSTEM_I_SYSTEM_FILE_H_INCLUDED_
#define _NBL_SYSTEM_I_SYSTEM_FILE_H_INCLUDED_


#include "nbl/system/IFile.h"


namespace nbl::system
{

class ISystemFile : public IFile
{
	protected:
		// the ISystem is the factory, so this stays protected
		explicit ISystemFile(
			core::smart_refctd_ptr<ISystem>&& _system,
			path&& _filename,
			const core::bitflag<E_CREATE_FLAGS> _flags,
			void* const _mappedPtr
		) : IFile(std::move(_filename),_flags,time_point_t()), m_system(std::move(_system)), m_mappedPtr(_mappedPtr) {}
		
		//
		inline void* getMappedPointer_impl() override {return m_mappedPtr;}
		inline const void* getMappedPointer_impl() const override {return m_mappedPtr;}
		
		//
		inline void unmappedRead(ISystem::future_t<size_t>& fut, void* buffer, size_t offset, size_t sizeToRead) override final
		{
			ISystem::SRequestParams_READ params;
			params.buffer = buffer;
			params.file = this;
			params.offset = offset;
			params.size = sizeToRead;
			m_system->m_dispatcher.request(&fut,params);
		}
		inline void unmappedWrite(ISystem::future_t<size_t>& fut, const void* buffer, size_t offset, size_t sizeToWrite) override final
		{
			ISystem::SRequestParams_WRITE params;
			params.buffer = buffer;
			params.file = this;
			params.offset = offset;
			params.size = sizeToWrite;
			m_system->m_dispatcher.request(&fut,params);
		}

		//
		friend struct ISystem::SRequestParams_READ;
		virtual size_t asyncRead(void* buffer, size_t offset, size_t sizeToRead) = 0;
		friend struct ISystem::SRequestParams_WRITE;
		virtual size_t asyncWrite(const void* buffer, size_t offset, size_t sizeToWrite) = 0;


		core::smart_refctd_ptr<ISystem> m_system;
		void* m_mappedPtr;
};

}

#endif
