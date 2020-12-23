#ifndef __NBL_C_WIN32_SURFACE_H_INCLUDED__
#define __NBL_C_WIN32_SURFACE_H_INCLUDED__

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace nbl {
namespace video
{

class ISurfaceWin32
{
public:
    struct SCreationParams
    {
        HINSTANCE hinstance;
        HWND hwnd;
    };

protected:
    ISurfaceWin32(SCreationParams&& params) : m_params(std::move(params)) {}

    SCreationParams m_params;
};

}
}

#endif