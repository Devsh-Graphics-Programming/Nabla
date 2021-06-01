#include "nbl/video/IOpenGL_FunctionTable.h"

thread_local char nbl::video::impl::g_NBL_GL_CALL_msg_buffer[4096];

namespace nbl {
namespace video
{

std::atomic_uint32_t IOpenGL_FunctionTable::s_guidGenerator = 0u;

}
}