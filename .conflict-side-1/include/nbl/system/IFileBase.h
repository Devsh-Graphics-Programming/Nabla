#ifndef _NBL_SYSTEM_I_FILE_BASE_H_INCLUDED_
#define _NBL_SYSTEM_I_FILE_BASE_H_INCLUDED_


#include "nbl/core/atomic.h"
#include "nbl/core/decl/smart_refctd_ptr.h"
#include "nbl/core/util/bitflag.h"

#include "nbl/system/path.h"

#include <filesystem>
#include <type_traits>


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

		//! Optional, if not present this means that the hash was not already precomputed for you.
		// Equivalent to calling `xxHash256(getMappedPointer(),getSize(),&retval.x)`
		// Only really available for built-in resources or some other files that had to be read in their entirety at some point.
		virtual inline std::optional<hlsl::uint64_t4> getPrecomputedHash() const {return {};}

		//!
		using time_point_t = std::chrono::utc_clock::time_point;
		virtual inline time_point_t getLastWriteTime() const;
		inline void setLastWriteTime(time_point_t tp=time_point_t::clock::now())
		{
			core::atomic_fetch_max(&m_modified,time_point_t::clock::now());
		}

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
		explicit IFileBase(path&& _filename, const core::bitflag<E_CREATE_FLAGS> _flags, const time_point_t _initialModified) :
			m_filename(std::move(_filename)), m_flags(_flags), m_modified(_initialModified) {}

	private:
		const path m_filename;
	protected:
		std::atomic<time_point_t> m_modified;
	private:
		const core::bitflag<E_CREATE_FLAGS> m_flags;
};

NBL_ENUM_ADD_BITWISE_OPERATORS(IFileBase::E_CREATE_FLAGS);

inline auto IFileBase::getLastWriteTime() const -> time_point_t
{
	// in theory should check if file is `coherent & (unlocked_someone_else_can_write | writemapped)`
	if (m_flags.hasFlags(ECF_WRITE|ECF_COHERENT))
		const_cast<IFileBase*>(this)->setLastWriteTime();
	return m_modified.load();
}

}

#endif
