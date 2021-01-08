#ifndef __NBL_I_WINDOW_H_INCLUDED__
#define __NBL_I_WINDOW_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"

namespace nbl {
namespace system
{

class IWindow : public core::IReferenceCounted
{
public:
    enum E_CREATE_FLAGS : uint32_t
    {
        ECF_FULLSCREEN = 1u<<0,
        ECF_HIDDEN = 1u<<1,
        ECF_BORDERLESS = 1u<<2,
        ECF_RESIZABLE = 1u<<3,
        ECF_MINIMIZED = 1u<<4,
        ECF_MAXIMIZED = 1u<<5,
        //! Forces mouse to stay inside the window
        ECF_MOUSE_CAPTURE = 1u<<6,
        //! Indicates whether the window is active or not
        ECF_INPUT_FOCUS = 1u<<7,
        //! Indicates whether mouse is hovering over the window even if the window is not active
        ECF_MOUSE_FOCUS = 1u<<8,
        ECF_ALWAYS_ON_TOP = 1u<<9
    };

    IWindow(uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags) :
        m_width(_w), m_height(_h), m_flags(_flags)
    {

    }

    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }

protected:
    virtual ~IWindow() = default;

    uint32_t m_width, m_height;
    E_CREATE_FLAGS m_flags;
};

}
}


#endif