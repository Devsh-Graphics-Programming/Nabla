#ifndef __NBL_E_API_TYPE_H_INCLUDED__
#define __NBL_E_API_TYPE_H_INCLUDED__

#include "nbl/core/declarations.h"
#include <cstdint>

namespace nbl::video
{

enum E_API_TYPE : uint32_t
{
    EAT_OPENGL,
    EAT_OPENGL_ES,
    EAT_VULKAN
};

}

#endif
