#ifndef __NBL_I_SURFACE_ANDROID_H_INCLUDED__
#define __NBL_I_SURFACE_ANDROID_H_INCLUDED__

#ifdef _NBL_PLATFORM_ANDROID_

#include <android/native_window.h>

namespace nbl {
namespace video
{

class ISurfaceAndroid
{
public:
    struct SCreationParams
    {
        struct ANativeWindow* anw;
    };

protected:
    explicit ISurfaceAndroid(SCreationParams&& params) : m_params(std::move(params))
    {

    }

    SCreationParams m_params;
};

}
}

#endif // _NBL_PLATFORM_ANDROID_

#endif