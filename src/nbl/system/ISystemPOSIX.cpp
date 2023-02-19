#include "nbl/system/ISystemPOSIX.h"
#include "nbl/system/CFilePOSIX.h"

#include "nbl/system/IFile.h"

using namespace nbl;
using namespace nbl::system;

#if defined(_NBL_PLATFORM_LINUX_) || defined (_NBL_PLATFORM_ANDROID_)

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

core::smart_refctd_ptr<ISystemFile> ISystemPOSIX::CCaller::createFile(const std::filesystem::path& filename, const core::bitflag<IFile::E_CREATE_FLAGS> flags)
{	
    const bool writeAccess = flags.value&IFile::ECF_WRITE;
	int createFlags = O_LARGEFILE|(writeAccess ? O_CREAT:0);
	switch (flags.value&IFile::ECF_READ_WRITE)
	{
		case IFile::ECF_READ:
			createFlags |= O_RDONLY;
			break;
		case IFile::ECF_WRITE:
			createFlags |= O_WRONLY;
			break;
		case IFile::ECF_READ_WRITE:
			createFlags |= O_RDWR;
			break;
		default:
			assert(false);
			break;
	}

	CFilePOSIX::native_file_handle_t _native = -1;
	auto filenameStream = filename.string();
	const char* name_c_str = filenameStream.c_str();
	// only create a new file if we're going to be writing
	if (writeAccess)
	{
		_native = creat(name_c_str, S_IRUSR | S_IRGRP | S_IROTH);//open(name_c_str, createFlags, S_IRUSR | S_IRGRP | S_IROTH);
	}
	else if (std::filesystem::exists(filename))
	{
		_native = open(name_c_str,createFlags);
	}
	else
		return nullptr;

	if (_native<0)
		return nullptr;

	// get size
	size_t _size;
	struct stat sb;
	if (stat(name_c_str,&sb) == -1)
	{
		close(_native);
		return nullptr;
	}
	else
		_size = sb.st_size;

	// map if needed
	void* _mappedPtr = nullptr;
	if (flags.value & IFile::ECF_MAPPABLE)
	{
		const int mappingFlags = ((flags.value&IFile::ECF_READ) ? PROT_READ:0)|(writeAccess ? PROT_WRITE:0);
		_mappedPtr = mmap((caddr_t)0, _size, mappingFlags, MAP_PRIVATE, _native, 0);
		if (_mappedPtr==MAP_FAILED)
		{
			close(_native);
			return nullptr;
		}
	}

	return core::make_smart_refctd_ptr<CFilePOSIX>(core::smart_refctd_ptr<ISystem>(m_system),path(filename),flags,_mappedPtr,_size,_native);
}
#endif
