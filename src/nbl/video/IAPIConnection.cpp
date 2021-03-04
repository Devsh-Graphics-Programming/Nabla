#include "nbl/video/IAPIConnection.h"

namespace nbl {
namespace video
{

// Functions defined in connections' .cpp files
// (i dont want to have all backends in single translation unit)
// as a result, if one wants to turn off compilation of whole backend, one can just remove corresponding API connection's .cpp from build
core::smart_refctd_ptr<IAPIConnection> createOpenGLConnection();
core::smart_refctd_ptr<IAPIConnection> createOpenGLESConnection();


core::smart_refctd_ptr<IAPIConnection> IAPIConnection::create(E_API_TYPE apiType, uint32_t appVer, const char* appName)
{
    switch (apiType)
    {
    case EAT_OPENGL:
        return createOpenGLConnection();
    case EAT_OPENGL_ES:
        return createOpenGLESConnection();
    //case EAT_VULKAN:
        //
    default:
        return nullptr;
    }
}

}
}