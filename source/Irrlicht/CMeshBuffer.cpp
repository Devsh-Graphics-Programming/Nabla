#include "IrrCompileConfig.h"
#include "IMeshBuffer.h"
#include "COpenGLExtensionHandler.h"

namespace irr
{
namespace scene
{

#ifdef _IRR_COMPILE_WITH_OPENGL_
IGPUMeshBuffer::IGPUMeshBuffer() : indexValMin(0), indexValMax(video::COpenGLExtensionHandler::MaxVertices-1) {}
#else
IGPUMeshBuffer::IGPUMeshBuffer() : indexValMin(0), indexValMax(0xffffffffu) {}
#endif // _IRR_COMPILE_WITH_OPENGL_


void IGPUMeshBuffer::setIndexRange(const uint32_t &minBeforeBaseVxAdd, const uint32_t &maxBeforeBaseVxAdd)
{
#ifdef defined(_DEBUG)&&defined( _IRR_COMPILE_WITH_OPENGL_)
    if (maxBeforeBaseVxAdd-minBeforeBaseVxAdd>=video::COpenGLExtensionHandler::MaxVertices)
        os::Printer::log("Too Many Vertices Per 1 DrawCall in MeshBuffer",ELL_ERROR);
#endif
    indexValMin = minBeforeBaseVxAdd;
    indexValMax = maxBeforeBaseVxAdd;
}


}
}
