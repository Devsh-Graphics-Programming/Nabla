#ifndef __NBL_I_WINDOW_XCB_H_INCLUDED__
#define __NBL_I_WINDOW_XCB_H_INCLUDED__

#ifdef _NBL_PLATFORM_LINUX_

#include "nbl/core/util/bitflag.h"

#include "nbl/ui/IWindow.h"
#include "nbl/ui/XCBConnection.h"

#include <xcb/xproto.h>

namespace nbl::ui
{

class NBL_API2 IWindowXCB : public IWindow
{
    protected:
        virtual ~IWindowXCB() = default;
        inline IWindowXCB(SCreationParams&& params) : IWindow(std::move(params)) {}

    public:
        using IWindow::IWindow;

        struct native_handle_t {
            xcb_window_t m_window;
            xcb_connection_t* m_connection;
        };

        virtual const native_handle_t* getNativeHandle() const = 0;

        virtual bool setWindowSize_impl(uint32_t width, uint32_t height) = 0;
        virtual bool setWindowPosition_impl(int32_t x, int32_t y) = 0;
        virtual bool setWindowRotation_impl(bool landscape) = 0;
        virtual bool setWindowVisible_impl(bool visible) = 0;
        virtual bool setWindowMaximized_impl(bool maximized) = 0;

        static XCBConnection::MotifWmHints fetchMotifMWHints(IWindow::E_CREATE_FLAGS flags) {
            core::bitflag<XCBConnection::MotifFlags> motifFlags(XCBConnection::MWM_HINTS_NONE);
            core::bitflag<XCBConnection::MotifFunctions> motifFunctions(XCBConnection::MWM_FUNC_NONE);
            core::bitflag<XCBConnection::MotifDecorations> motifDecorations(XCBConnection::MWM_DECOR_NONE);
            motifFlags |= XCBConnection::MWM_HINTS_DECORATIONS;

            if (flags & IWindow::ECF_BORDERLESS) {
                motifDecorations |= XCBConnection::MWM_DECOR_ALL;
            } else {
                motifDecorations |= XCBConnection::MWM_DECOR_BORDER;
                motifDecorations |= XCBConnection::MWM_DECOR_RESIZEH;
                motifDecorations |= XCBConnection::MWM_DECOR_TITLE;

                // minimize button
                if(flags & IWindow::ECF_MINIMIZED) {
                    motifDecorations |= XCBConnection::MWM_DECOR_MINIMIZE;
                    motifFunctions |= XCBConnection::MWM_FUNC_MINIMIZE;
                }
                
                // maximize button
                if(flags & IWindow::ECF_MAXIMIZED) {
                    motifDecorations |= XCBConnection::MWM_DECOR_MAXIMIZE;
                    motifFunctions |= XCBConnection::MWM_FUNC_MAXIMIZE;
                }

                // close button
                motifFunctions |= XCBConnection::MWM_FUNC_CLOSE;
            }

            if(motifFunctions.value != XCBConnection::MWM_FUNC_NONE) {
                motifFlags |= XCBConnection::MWM_HINTS_FUNCTIONS;
                motifFunctions |= XCBConnection::MWM_FUNC_RESIZE;
                motifFunctions |= XCBConnection::MWM_FUNC_MOVE;
            } else {
                motifFunctions = XCBConnection::MWM_FUNC_ALL;
            }

            XCBConnection::MotifWmHints hints;
            hints.flags = motifFlags.value;
            hints.functions = motifFunctions.value;
            hints.decorations = motifDecorations.value;
            hints.input_mode = 0;
            hints.status = 0;
            return hints;

        }
		
};

}

#endif

#endif