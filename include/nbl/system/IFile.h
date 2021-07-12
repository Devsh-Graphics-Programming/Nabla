#ifndef __NBL_I_FILE_H_INCLUDED__
#define __NBL_I_FILE_H_INCLUDED__

#include <filesystem>
#include <type_traits>
#include "nbl/core/IReferenceCounted.h"

namespace nbl {
namespace system
{
class IFile : public core::IReferenceCounted
{
	// Don't like this approach, gonna stack up many forward declarations, @criss any suggestions?
	// I also had an idea to make interfaces public in the implementations of IFile
	// and static_cast files before calling, but that would invalidate any other IFile descendants calls
	friend class CSystemCallerWin32;
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

	explicit IFile(std::underlying_type_t<E_CREATE_FLAGS> _flags) : m_flags(_flags) {}

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
protected: 
	virtual int32_t read(void* buffer, size_t offset, size_t sizeToRead) = 0;
	virtual int32_t write(const void* buffer, size_t offset, size_t sizeToWrite) = 0;
protected:
	std::underlying_type_t<E_CREATE_FLAGS> m_flags;
};

}
}

#endif
