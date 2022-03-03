#ifndef _NBL_SYSTEM_I_FILE_BASE_H_INCLUDED_
#define _NBL_SYSTEM_I_FILE_BASE_H_INCLUDED_

#include "nbl/core/decl/smart_refctd_ptr.h"
#include "nbl/core/util/bitflag.h"

#include <filesystem>
#include <type_traits>

#include "nbl/system/path.h"

namespace nbl::system
{

class IFileBase : public core::IReferenceCounted
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

		// static utility, needed because std::filesystem won't flatten virtual paths (those that don't exist)
		static path flattenFilename(const path& p);

	protected:
		virtual void* getMappedPointer_impl() = 0;
		virtual const void* getMappedPointer_impl() const = 0;

		// this is an abstract interface class so this stays protected
		explicit IFileBase(path&& _filename, const core::bitflag<E_CREATE_FLAGS> _flags) : m_filename(std::move(_filename)), m_flags(_flags) {}
		explicit IFileBase(const path& _filename, const core::bitflag<E_CREATE_FLAGS> _flags) : m_filename(_filename), m_flags(_flags) {}

	private:
		path m_filename;
		core::bitflag<E_CREATE_FLAGS> m_flags;
};

}

#endif
