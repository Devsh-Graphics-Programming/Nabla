#ifndef __NBL_I_WINDOW_H_INCLUDED__
#define __NBL_I_WINDOW_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/system/ISystem.h"
#include "nbl/ui/IClipboardManager.h"

namespace nbl {
namespace ui
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
        ECF_ALWAYS_ON_TOP = 1u<<9,

        ECF_NONE = 0
    };

    inline bool isFullscreen()      { return (m_flags & ECF_FULLSCREEN); }
    inline bool isHidden()          { return (m_flags & ECF_HIDDEN); }
    inline bool isBorderless()      { return (m_flags & ECF_BORDERLESS); }
    inline bool isResizable()       { return (m_flags & ECF_RESIZABLE); }
    inline bool isMinimized()       { return (m_flags & ECF_MINIMIZED); }
    inline bool isMaximized()       { return (m_flags & ECF_MAXIMIZED); }
    inline bool hasMouseCaptured()  { return (m_flags & ECF_MOUSE_CAPTURE); }
    inline bool hasInputFocus()     { return (m_flags & ECF_INPUT_FOCUS); }
    inline bool hasMouseFocus()     { return (m_flags & ECF_MOUSE_FOCUS); }
    inline bool isAlwaysOnTop()     { return (m_flags & ECF_ALWAYS_ON_TOP); }

    inline uint32_t getWidth() const { return m_width; }
    inline uint32_t getHeight() const { return m_height; }

    virtual IClipboardManager* getClipboardManager() = 0;

protected:
    IWindow(core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t _w = 0u, uint32_t _h = 0u, E_CREATE_FLAGS _flags = static_cast<E_CREATE_FLAGS>(0)) :
        m_sys(std::move(sys)), m_width(_w), m_height(_h), m_flags(_flags)
    {

    }

    virtual ~IWindow() = default;

    core::smart_refctd_ptr<system::ISystem> m_sys;
    uint32_t m_width = 0u, m_height = 0u;
    E_CREATE_FLAGS m_flags = static_cast<E_CREATE_FLAGS>(0);
};

}
}


#endif