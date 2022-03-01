#ifndef _NBL_SYSTEM_I_FILE_H_INCLUDED_
#define _NBL_SYSTEM_I_FILE_H_INCLUDED_

#include "nbl/core/decl/smart_refctd_ptr.h"
#include "nbl/core/util/bitflag.h"

#include "nbl/system/path.h"

#include <filesystem>
#include <type_traits>

namespace nbl::system
{

class ISystem;

// TODO:
//namespace impl
//{
template<typename T>
class future;
//}

class IFile : public core::IReferenceCounted
{
		// TODO: how many of these friends do we actually need?
		friend class ISystemCaller;
		friend class ISystem;
		friend class IFileArchive;

	public:
		enum E_CREATE_FLAGS : uint32_t
		{
			ECF_READ = 0b0001,
			ECF_WRITE = 0b0010,
			ECF_READ_WRITE = 0b0011,
			ECF_MAPPABLE = 0b0100,
			//! Implies ECF_MAPPABLE
			ECF_COHERENT = 0b1100
		};

		//! Get size of file.
		/** \return Size of the file in bytes. */
		virtual size_t getSize() const = 0;

		//! Get name of file.
		/** \return File name as zero terminated character string. */
		inline const path& getFileName() const { return m_filename; }

		//
		inline E_CREATE_FLAGS getFlags() const { return m_flags.value; }

		//
		inline const void* getMappedPointer() const
		{
			if (m_flags.value&ECF_READ)
				return getMappedPointer_impl();
			return nullptr;
		}
		void* getMappedPointer()
		{
			if (m_flags.value&ECF_WRITE)
				return getMappedPointer_impl();
			return nullptr;
		}
	
		//
		inline bool isMappingCoherent() const
		{
			return m_flags.value&ECF_COHERENT;
		}

		// TODO: make this virtual, so the IFileView does not need to keep ref to ISystem
		void read(future<size_t>& fut, void* buffer, size_t offset, size_t sizeToRead);
		void write(future<size_t>& fut, const void* buffer, size_t offset, size_t sizeToWrite);

		/* TODO: future utility
		struct success
		{
			public:
				success() = default;
				~success() = default;
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
				future<size_t> m_internalFuture;
				size_t sizeToProcess;
		};
		void read(success& fut, void* buffer, size_t offset, size_t sizeToRead)
		{
			read(fut.m_internalFuture,buffer,offset,sizeToRead);
			fut.sizeToProcess = sizeToRead;
		}
		void write(success& fut, const void* buffer, size_t offset, size_t sizeToWrite)
		{
			write(fut.m_internalFuture,buffer,offset,sizeToWrite);
			fut.sizeToProcess = sizeToWrite;
		}
		*/

		static path flattenFilename(const path& p);

	protected:
		// TODO: docs
		virtual size_t read_impl(void* buffer, size_t offset, size_t sizeToRead) = 0;
		virtual size_t write_impl(const void* buffer, size_t offset, size_t sizeToWrite) = 0;

		virtual void* getMappedPointer_impl() = 0;
		virtual const void* getMappedPointer_impl() const = 0;

		// the ISystem is the factory, so this stays protected
		explicit IFile(core::smart_refctd_ptr<ISystem>&& _system, const path& _filename, core::bitflag<E_CREATE_FLAGS> _flags);

		core::smart_refctd_ptr<ISystem> m_system;
		core::bitflag<E_CREATE_FLAGS> m_flags;

	private:
		path m_filename;
};

}

#endif
