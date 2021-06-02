#include "nbl/video/IOpenGL_FunctionTable.h"

namespace nbl {
namespace video
{

std::atomic_uint32_t IOpenGL_FunctionTable::s_guidGenerator = 0u;

namespace impl
{
    thread_local char g_NBL_GL_CALL_msg_buffer[4096];
}

}
}
