#ifndef __NBL_I_FILE_H_INCLUDED__
#define __NBL_I_FILE_H_INCLUDED__

#include "nbl/core/decl/smart_refctd_ptr.h"

#include <filesystem>
#include <type_traits>

namespace nbl::system
{

class ISystem;

class IFile : public core::IReferenceCounted
{
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
		virtual const std::filesystem::path& getFileName() const = 0;

		E_CREATE_FLAGS getFlags() const { return static_cast<E_CREATE_FLAGS>(m_flags); }

		virtual void* getMappedPointer() = 0;
		virtual const void* getMappedPointer() const = 0;
	

		bool isMappingCoherent() const
		{
			return (m_flags & ECF_COHERENT) == ECF_COHERENT;
		}

		// TODO: make the `ISystem` methods protected instead 
		virtual int32_t read(void* buffer, size_t offset, size_t sizeToRead) = 0;
		virtual int32_t write(const void* buffer, size_t offset, size_t sizeToWrite) = 0;

	protected:
		// the ISystem is the factory, so this starys protected
		explicit IFile(core::smart_refctd_ptr<ISystem>&& _system, std::underlying_type_t<E_CREATE_FLAGS> _flags); // TODO: implement in IFile.cpp

		const core::smart_refctd_ptr<ISystem> m_system;
		std::underlying_type_t<E_CREATE_FLAGS> m_flags;
};

}

#endif
