#ifndef __NBL_I_OPENGL__LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_I_OPENGL__LOGICAL_DEVICE_H_INCLUDED__

#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/CEGL.h"
#include "nbl/system/IThreadHandler.h"
#include "nbl/core/alloc/PoolAddressAllocator.h"

namespace nbl {
namespace video
{

// Common base for GL and GLES logical devices
// All COpenGL* objects (buffers, images, views...) will keep pointer of this type (just to be able to request destruction in destructor though)
// Implementation of both GL and GLES is the same code (see COpenGL_LogicalDevice) thanks to IOpenGL_FunctionTable abstraction layer
class IOpenGL_LogicalDevice : public ILogicalDevice
{
protected:
    enum E_REQUEST_TYPE
    {
        ERT_BUFFER_CREATE,
        ERT_BUFFER_DESTROY,

        ERT_BUFFER_VIEW_CREATE,
        ERT_BUFFER_VIEW_DESTROY,

        ERT_IMAGE_CREATE,
        ERT_IMAGE_DESTROY,

        ERT_IMAGE_VIEW_CREATE,
        ERT_IMAGE_VIEW_DESTROY,

        ERT_FRAMEBUFFER_CREATE,
        ERT_FRAMEBUFFER_DESTROY,

        ERT_SWAPCHAIN_CREATE,
        ERT_SWAPCHAIN_DESTROY,

        //....
    };

    struct SRequest
    {
        E_REQUEST_TYPE type;
        // TODO uncomment below later
        //std::variant<All,Possible,Types,Of,Parameter,Structs,For,Requests>

        // only relevant in case of CREATE requests
        core::smart_refctd_ptr<core::IReferenceCounted> retval;
        // wait on this for `retval` to become valid
        std::condition_variable cvar;
    };

    template <typename FunctionTableType>
    class CThreadHandler : public system::IThreadHandler<FunctionTableType>
    {
        constexpr static inline uint32_t MaxAwaitingRequests = 256u;

        using request_alctr_t = core::PoolAddressAllocator<uint32_t>; // TODO try with circualr buffer instead
        SRequest request_pool[MaxAwaitingRequests];
        request_alctr_t request_alctr;

    public:
        // T must be one of request parameter structs
        template <typename T>
        SRequest& request(const T& params)
        {
            auto raii_handler = createRAIIDisptachHandler();

            const uint32_t r_id = request_alctr.alloc_addr(1u, 1u);

            return request_pool[r_id];
        }
    };

public:
    IOpenGL_LogicalDevice(const egl::CEGL* _egl, EGLConfig config, EGLint major, EGLint minor, const SCreationParams& params) : ILogicalDevice(params)
    {

    }
};

}
}

#endif
