#ifndef __NBL_DEBUG_H_INCLUDED__
#define __NBL_DEBUG_H_INCLUDED__

#include "nbl/core/decl/compile_config.h"
#include <cstdint>

namespace nbl {
namespace video
{

enum E_DEBUG_MESSAGE_SEVERITY : uint32_t
{
    EDMS_VERBOSE = 0x00000001,
    EDMS_INFO = 0x00000010,
    EDMS_WARNING = 0x00000100,
    EDMS_ERROR = 0x00001000
};
enum E_DEBUG_MESSAGE_TYPE : uint32_t
{
    EDMT_GENERAL = 0x00000001,
    EDMT_VALIDATION = 0x00000002,
    EDMT_PERFORMANCE = 0x00000004
};

struct SDebugCallback
{
    using callback_func_t = void(*)(E_DEBUG_MESSAGE_SEVERITY severity, E_DEBUG_MESSAGE_TYPE type, const char* message, void* userData);

    callback_func_t callback;
    void* userData;
};

}
}

#endif
