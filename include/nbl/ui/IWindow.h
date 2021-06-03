#ifndef __NBL_I_WINDOW_H_INCLUDED__
#define __NBL_I_WINDOW_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/system/ISystem.h"
#include "nbl/ui/IClipboardManager.h"
#include "nbl/ui/IInputEventChannel.h"
#include <type_traits>

namespace nbl {
namespace ui
{

class IWindow : public core::IReferenceCounted
{
public:
    enum E_CREATE_FLAGS : uint32_t
    {
        ECF_FULLSCREEN = 1u << 0,
        ECF_HIDDEN = 1u << 1,
        ECF_BORDERLESS = 1u << 2,
        ECF_RESIZABLE = 1u << 3,
        ECF_MINIMIZED = 1u << 4,
        ECF_MAXIMIZED = 1u << 5,
        //! Forces mouse to stay inside the window
        ECF_MOUSE_CAPTURE = 1u << 6,
        //! Indicates whether the window is active or not
        ECF_INPUT_FOCUS = 1u << 7,
        //! Indicates whether mouse is hovering over the window even if the window is not active
        ECF_MOUSE_FOCUS = 1u << 8,
        ECF_ALWAYS_ON_TOP = 1u << 9,

        ECF_NONE = 0
    };

    struct SCreationParams
    {
        //IWindow(core::smart_refctd_ptr<IEventCallback>&& _cb, core::smart_refctd_ptr<system::ISystem>&& _sys, uint32_t _w = 0u, uint32_t _h = 0u, E_CREATE_FLAGS _flags = static_cast<E_CREATE_FLAGS>(0)) :
        core::smart_refctd_ptr<IEventCallback> callback;
        core::smart_refctd_ptr<system::ISystem> system;
        int32_t x, y;
        uint32_t width = 0u, height = 0u;
        E_CREATE_FLAGS flags = static_cast<E_CREATE_FLAGS>(0);
        uint32_t eventChannelCapacityLog2[IInputEventChannel::ET_COUNT];
        std::string windowCaption;
    };

    friend class IEventCallback;
    class IEventCallback : public core::IReferenceCounted
    {
    public:
        void onWindowShown(IWindow* w) 
        {
            w->m_flags &= (~ECF_HIDDEN);
            onWindowShown_impl();
        }
        void onWindowHidden(IWindow* w)
        {
            w->m_flags |= ECF_HIDDEN;
            onWindowHidden_impl();
        }
        void onWindowMoved(IWindow* w, int x, int y)
        {
            onWindowMoved_impl(x, y);
        }
        void onWindowResized(IWindow* w, uint32_t width, uint32_t height)
        {
            w->m_width = width;
            w->m_height = height;
            onWindowResized_impl(width, height);
        }
        void onWindowRotated(IWindow* w)
        {
            onWindowRotated_impl();
        }
        void onWindowMinimized(IWindow* w)
        {
            w->m_flags |= ECF_MINIMIZED;
            w->m_flags &= (~ECF_MAXIMIZED);
            onWindowMinimized_impl();
        }
        void onWindowMaximized(IWindow* w)
        {
            w->m_flags |= ECF_MAXIMIZED;
            w->m_flags &= (~ECF_MINIMIZED);
            onWindowMaximized_impl();
        }
        void onGainedMouseFocus(IWindow* w)
        {
            onGainedMouseFocus_impl();
        }
        void onLostMouseFocus(IWindow* w)
        {
            onLostMouseFocus_impl();
        }
        void onGainedKeyboardFocus(IWindow* w)
        {
            onGainedKeyboardFocus_impl();
        }
        void onLostKeyboardFocus(IWindow* w)
        {
            onLostKeyboardFocus_impl();
        }

        void onMouseConnected(IWindow* w, core::smart_refctd_ptr<IMouseEventChannel>&& mch)
        {
            onMouseConnected_impl(std::move(mch));
        }
        void onMouseDisconnected(IWindow* w, IMouseEventChannel* mch)
        {
            onMouseDisconnected_impl(mch);
        }
        void onKeyboardConnected(IWindow* w, core::smart_refctd_ptr<IKeyboardEventChannel>&& kbch)
        {
            onKeyboardConnected_impl(std::move(kbch));
        }
        void onKeyboardDisconnected(IWindow* w, IKeyboardEventChannel* kbch)
        {
            onKeyboardDisconnected_impl(kbch);
        }

    protected:
        virtual void onWindowShown_impl() {}
        virtual void onWindowHidden_impl() {}
        virtual void onWindowMoved_impl(int x, int y) {}
        virtual void onWindowResized_impl(uint32_t w, uint32_t h) {}
        virtual void onWindowRotated_impl() {}
        virtual void onWindowMinimized_impl() {}
        virtual void onWindowMaximized_impl() {}
        virtual void onGainedMouseFocus_impl() {}
        virtual void onLostMouseFocus_impl() {}
        virtual void onGainedKeyboardFocus_impl() {}
        virtual void onLostKeyboardFocus_impl() {}

        virtual void onMouseConnected_impl(core::smart_refctd_ptr<IMouseEventChannel>&& mch) {}
        virtual void onMouseDisconnected_impl(IMouseEventChannel* mch) {}
        virtual void onKeyboardConnected_impl(core::smart_refctd_ptr<IKeyboardEventChannel>&& kbch) {}
        virtual void onKeyboardDisconnected_impl(IKeyboardEventChannel* mch) {}
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

    // Not sure where this should be. Obviously not here
    char keyCodeToChar(E_KEY_CODE code)
    {
        char result = 0;
        switch (code)
        {
        case EKC_0: [[fallthrough]];
        case EKC_NUMPAD_0: result = '0';
        case EKC_1: [[fallthrough]];
        case EKC_NUMPAD_1: result = '1';
        case EKC_2: [[fallthrough]];
        case EKC_NUMPAD_2: result = '2';
        case EKC_3: [[fallthrough]];
        case EKC_NUMPAD_3: result = '3';
        case EKC_4: [[fallthrough]];
        case EKC_NUMPAD_4: result = '4';
        case EKC_5: [[fallthrough]];
        case EKC_NUMPAD_5: result = '5';
        case EKC_6: [[fallthrough]];
        case EKC_NUMPAD_6: result = '6';
        case EKC_7: [[fallthrough]];
        case EKC_NUMPAD_7: result = '7';
        case EKC_8: [[fallthrough]];
        case EKC_NUMPAD_8: result = '8';
        case EKC_9: [[fallthrough]];
        case EKC_NUMPAD_9: result = '9';

        case EKC_A: result = 'a';
        case EKC_B: result = 'b';
        case EKC_C: result = 'c';
        case EKC_D: result = 'd';
        case EKC_E: result = 'e';
        case EKC_F: result = 'f';
        case EKC_G: result = 'g';
        case EKC_H: result = 'h';
        case EKC_I: result = 'i';
        case EKC_J: result = 'j';
        case EKC_K: result = 'k';
        case EKC_L: result = 'l';
        case EKC_M: result = 'm';
        case EKC_N: result = 'n';
        case EKC_O: result = 'o';
        case EKC_P: result = 'p';
        case EKC_Q: result = 'q';
        case EKC_R: result = 'r';
        case EKC_S: result = 's';
        case EKC_T: result = 't';
        case EKC_U: result = 'u';
        case EKC_V: result = 'v';
        case EKC_W: result = 'w';
        case EKC_X: result = 'x';
        case EKC_Y: result = 'y';
        case EKC_Z: result = 'x';
        
        case EKC_TAB: result = '\t';
        case EKC_ENTER: result = '\n';
        case EKC_SPACE: result = ' ';
        }
        return result;
    }
    virtual IClipboardManager* getClipboardManager() = 0;
    IEventCallback* getEventCallback() const { return m_cb.get(); }

protected:
    // TODO need to update constructors of all derived CWindow* classes
    IWindow(SCreationParams&& params) :
        m_cb(std::move(params.callback)), m_sys(std::move(params.system)), m_width(params.width), m_height(params.height), m_flags(params.flags)
    {

    }

    virtual ~IWindow() = default;

    core::smart_refctd_ptr<IEventCallback> m_cb;
    core::smart_refctd_ptr<system::ISystem> m_sys;
    uint32_t m_width = 0u, m_height = 0u;
    // TODO add x,y window position to state?
    std::underlying_type_t<E_CREATE_FLAGS> m_flags = 0u;
};

}
}


#endif