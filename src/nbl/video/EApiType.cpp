#include "nbl/video/EApiType.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace nbl::video
{

bool CloseExternalHandle(external_handle_t handle)
{
#ifdef _WIN32
	return CloseHandle(handle);
#else
	return close(handle)==0;
#endif
}

external_handle_t DuplicateExternalHandle(external_handle_t handle)
{
#ifdef _WIN32
	HANDLE duplicated = ExternalHandleNull;

	const HANDLE process = GetCurrentProcess();
	if (!DuplicateHandle(process,handle,process,&duplicated,GENERIC_ALL,0,DUPLICATE_SAME_ACCESS))
		return ExternalHandleNull;

	return duplicated;
#else
	return dup(handle);
#endif
}

}
