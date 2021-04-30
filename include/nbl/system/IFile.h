#ifndef __NBL_I_FILE_H_INCLUDED__
#define __NBL_I_FILE_H_INCLUDED__

#include <filesystem>
#include "nbl/core/IReferenceCounted.h"

namespace nbl {
namespace system
{

class IFile : public core::IReferenceCounted
{
public:
	enum E_CREATE_FLAGS : uint32_t
	{
		ECF_READ = 0b01,
		ECF_WRITE = 0b10,
		ECF_READ_WRITE = 0b11
	};

	explicit IFile(E_CREATE_FLAGS _flags) : m_flags(_flags) {}

	//! Get size of file.
	/** \return Size of the file in bytes. */
	virtual size_t getSize() const = 0;

	//! Get name of file.
	/** \return File name as zero terminated character string. */
	virtual const std::filesystem::path& getFileName() const = 0;

	E_CREATE_FLAGS getFlags() const { return m_flags; }

protected:
	E_CREATE_FLAGS m_flags;
};

}
}

#endif
