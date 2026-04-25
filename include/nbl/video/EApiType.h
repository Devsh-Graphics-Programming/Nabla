#ifndef __NBL_E_API_TYPE_H_INCLUDED__
#define __NBL_E_API_TYPE_H_INCLUDED__

#include "nbl/core/declarations.h"
#include <cstdint>

namespace nbl::video
{

enum E_API_TYPE : uint32_t
{
    EAT_VULKAN,
    //EAT_WEBGPU
};

// TODO(kevinyu): Should I move this type and functions to its own file?
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
    return (close(handle) == 0);
#endif
}

inline external_handle_t DuplicateExternalHandle(external_handle_t handle)
{
#ifdef _WIN32
    HANDLE re = ExternalHandleNull;

    const HANDLE cur = GetCurrentProcess();
    if (!DuplicateHandle(cur, handle, cur, &re, GENERIC_ALL, 0, DUPLICATE_SAME_ACCESS))
        return ExternalHandleNull;

    return re;
#else
    return dup(handle);
#endif
}

}

#endif
