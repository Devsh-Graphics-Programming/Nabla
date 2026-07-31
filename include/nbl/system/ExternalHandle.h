#ifndef __NBL_EXTERNAL_HANDLE_INCLUDED__
#define __NBL_EXTERNAL_HANDLE_INCLUDED__

#ifdef _WIN32
	#ifndef WIN32_LEAN_AND_MEAN
		#define WIN32_LEAN_AND_MEAN
	#endif
	#include <windows.h>
#else
	#include <unistd.h>
#endif

namespace nbl::system 
{

using external_handle_t =
#ifdef _WIN32
  void*
#else
  int
#endif
  ;

#ifdef _WIN32
constexpr external_handle_t ExternalHandleNull = nullptr;
#else
constexpr external_handle_t ExternalHandleNull = -1;
#endif

inline bool CloseExternalHandle(external_handle_t handle)
{
#ifdef _WIN32
  return CloseHandle(handle);
#else
  return close(handle) == 0;
#endif
}

inline external_handle_t DuplicateExternalHandle(external_handle_t handle)
{
#ifdef _WIN32
  HANDLE duplicated = ExternalHandleNull;

  const HANDLE process = GetCurrentProcess();
  if (!DuplicateHandle(process, handle, process, &duplicated, GENERIC_ALL, 0, DUPLICATE_SAME_ACCESS))
    return ExternalHandleNull;

  return duplicated;
#else
  return dup(handle);
#endif
}

}

#endif
