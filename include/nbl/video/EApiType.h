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

NBL_API2 bool CloseExternalHandle(external_handle_t handle);
NBL_API2 external_handle_t DuplicateExternalHandle(external_handle_t handle);

}

#endif
