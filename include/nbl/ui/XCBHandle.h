#ifndef C_XCB_CONNECTION_XCB
#define C_XCB_CONNECTION_XCB

#ifdef _NBL_PLATFORM_LINUX_

#include "nbl/core/decl/Types.h"
#include "nbl/core/string/StringLiteral.h"
#include "nbl/core/decl/smart_refctd_ptr.h"
#include "nbl/ui/IWindowManagerXCB.h"

#include <xcb/xcb.h>
#include <xcb/xcb_icccm.h>
#include <xcb/xproto.h>

#include <cstdint>

namespace nbl::ui::xcb
{
    class XCBHandle;

	enum MotifFlags: uint32_t {
        MWM_HINTS_NONE = 0,
        MWM_HINTS_FUNCTIONS = (1L << 0),
        MWM_HINTS_DECORATIONS = (1L << 1),
        MWM_HINTS_INPUT_MODE = (1L << 2),
        MWM_HINTS_STATUS = (1L << 3),
    };

    enum MotifFunctions: uint32_t {
        MWM_FUNC_NONE = 0,
        MWM_FUNC_ALL      = (1L << 0),
        MWM_FUNC_RESIZE   = (1L << 1),
        MWM_FUNC_MOVE     = (1L << 2),
        MWM_FUNC_MINIMIZE = (1L << 3),
        MWM_FUNC_MAXIMIZE = (1L << 4),
        MWM_FUNC_CLOSE    = (1L << 5),
    };

    enum MotifDecorations: uint32_t {
        MWM_DECOR_NONE = 0,
        MWM_DECOR_ALL      = (1L << 0),
        MWM_DECOR_BORDER   = (1L << 1),
        MWM_DECOR_RESIZEH  = (1L << 2),
        MWM_DECOR_TITLE    = (1L << 3),
        MWM_DECOR_MENU     = (1L << 4),
        MWM_DECOR_MINIMIZE = (1L << 5),
        MWM_DECOR_MAXIMIZE = (1L << 6),
    };

	// insane magic in xcb for window hinting good luck finding documentation
    // https://fossies.org/linux/motif/lib/Xm/MwmUtil.h
    struct MotifWmHints {
        MotifFlags flags = MotifFlags::MWM_HINTS_NONE;
        MotifFunctions functions = MotifFunctions::MWM_FUNC_NONE;
        MotifDecorations decorations = MotifDecorations::MWM_DECOR_NONE;
        uint32_t input_mode = 0; // unused
        uint32_t status = 0; // unused
    };

    inline MotifWmHints createFlagsToMotifWmHints(IWindow::E_CREATE_FLAGS flags) {
        core::bitflag<MotifFlags> motifFlags(MWM_HINTS_NONE);
        core::bitflag<MotifFunctions> motifFunctions(MWM_FUNC_NONE);
        core::bitflag<MotifDecorations> motifDecorations(MWM_DECOR_NONE);
        motifFlags |= MWM_HINTS_DECORATIONS;

        if (flags & IWindow::ECF_BORDERLESS) {
            motifDecorations |= MWM_DECOR_ALL;
        } else {
            motifDecorations |= MWM_DECOR_BORDER;
            motifDecorations |= MWM_DECOR_RESIZEH;
            motifDecorations |= MWM_DECOR_TITLE;

            // minimize button
            if(flags & IWindow::ECF_MINIMIZED) {
                motifDecorations |= MWM_DECOR_MINIMIZE;
                motifFunctions |= MWM_FUNC_MINIMIZE;
            }
            
            // maximize button
            if(flags & IWindow::ECF_MAXIMIZED) {
                motifDecorations |= MWM_DECOR_MAXIMIZE;
                motifFunctions |= MWM_FUNC_MAXIMIZE;
            }

            // close button
            motifFunctions |= MWM_FUNC_CLOSE;
        }

        if(motifFunctions.value != MWM_FUNC_NONE) {
            motifFlags |= MWM_HINTS_FUNCTIONS;
            motifFunctions |= MWM_FUNC_RESIZE;
            motifFunctions |= MWM_FUNC_MOVE;
        } else {
            motifFunctions = MWM_FUNC_ALL;
        }

        MotifWmHints hints;
        hints.flags = motifFlags.value;
        hints.functions = motifFunctions.value;
        hints.decorations = motifDecorations.value;
        hints.input_mode = 0;
        hints.status = 0;
        return hints;
    }

    class XCBHandle final : public core::IReferenceCounted {
		public:
            template<core::StringLiteral Name> 
            struct AtomToken {
            private:
                AtomToken(XCBHandle& handle) : m_handle(handle) {}
                XCBHandle& m_handle;
                xcb_atom_t m_token = 0;
                bool fetched = false;
            public:
                inline xcb_atom_t fetch(bool only_if_exists = true, 
                    bool forced = false) {
                    const auto& xcb = m_handle.getXcbFunctionTable();
                    if(fetched && !forced) {
                        return m_token;
                    }
                    fetched = true;
                    size_t size = Name.size() - 1; // -1 to remove the null terminator
                    xcb_intern_atom_cookie_t cookie = xcb.pxcb_intern_atom(m_handle, only_if_exists, size, Name.value);
                    if(xcb_intern_atom_reply_t* reply = xcb.pxcb_intern_atom_reply(m_handle, cookie, nullptr)) {
                        m_token = reply->atom;
                        free(reply);
                        return m_token;
                    }
                    return m_token;
                }

                friend class XCBHandle;
            };
            
			XCBHandle(core::smart_refctd_ptr<IWindowManagerXCB>&& windowManager):
				m_windowManager(std::move(windowManager)) {
				const auto& xcb = m_windowManager->getXcbFunctionTable();
				m_connection = xcb.pxcb_connect(nullptr, nullptr);
			}
			~XCBHandle() {
				if(m_connection) {
                    const auto& xcb = m_windowManager->getXcbFunctionTable();
					xcb.pxcb_disconnect(m_connection);
				}
			}

			inline IWindowManagerXCB* windowManager() const { return m_windowManager.get(); }
			inline const IWindowManagerXCB::Xcb& getXcbFunctionTable() const { return m_windowManager->getXcbFunctionTable(); }
			inline const IWindowManagerXCB::XcbIcccm& getXcbIcccmFunctionTable() const { return m_windowManager->getXcbIcccmFunctionTable(); }
			inline operator xcb_connection_t*() { return m_connection; }
            inline xcb_connection_t* getNativeHandle() { return m_connection; }


            AtomToken<core::StringLiteral("WM_DELETE_WINDOW")> WM_DELETE_WINDOW{*this};
            AtomToken<core::StringLiteral("WM_PROTOCOLS")> WM_PROTOCOLS{*this};
            AtomToken<core::StringLiteral("_NET_WM_PING")> _NET_WM_PING{*this};
            
            AtomToken<core::StringLiteral("_NET_WM_STATE_MAXIMIZED_VERT")> _NET_WM_STATE_MAXIMIZED_VERT{*this};
            AtomToken<core::StringLiteral("_NET_WM_STATE_MAXIMIZED_HORZ")> _NET_WM_STATE_MAXIMIZED_HORZ{*this};
            AtomToken<core::StringLiteral("_NET_WM_STATE_FULLSCREEN")> _NET_WM_STATE_FULLSCREEN{*this};
            AtomToken<core::StringLiteral("_NET_WM_STATE")> _NET_WM_STATE{*this};
            AtomToken<core::StringLiteral("_MOTIF_WM_HINTS")> _MOTIF_WM_HINTS{*this};
            AtomToken<core::StringLiteral("NET_WM_STATE_ABOVE")> NET_WM_STATE_ABOVE{*this};

            AtomToken<core::StringLiteral("CLIPBOARD")> m_CLIPBOARD{*this};
            AtomToken<core::StringLiteral("TARGETS")> m_TARGETS{*this};
            AtomToken<core::StringLiteral("INCR")> m_INCR{*this};

            AtomToken<core::StringLiteral("UTF8_STRING")> m_formatUTF8_0{*this};
            AtomToken<core::StringLiteral("text/plain;charset=utf-8")> m_formatUTF8_1{*this};
            AtomToken<core::StringLiteral("text/plain;charset=UTF-8")> m_formatUTF8_2{*this};
            AtomToken<core::StringLiteral("GTK_TEXT_BUFFER_CONTENTS")> m_formatGTK{*this};
            AtomToken<core::StringLiteral("STRING")> m_formatString{*this};
            AtomToken<core::StringLiteral("TEXT")> m_formatText{*this};
            AtomToken<core::StringLiteral("text/plain")> m_formatTextPlain{*this};
		private:
			core::smart_refctd_ptr<IWindowManagerXCB> m_windowManager;
			xcb_connection_t* m_connection = nullptr;
		};

        inline void setMotifWmHints(XCBHandle& handle, xcb_window_t window, const MotifWmHints& hint) {
            auto& xcb = handle.getXcbFunctionTable();

            auto atomHint = handle._MOTIF_WM_HINTS.fetch();
            if(hint.flags != MotifFlags::MWM_HINTS_NONE) {
                xcb.pxcb_change_property(handle.getNativeHandle(), XCB_PROP_MODE_REPLACE, window, 
                    atomHint, 
                    atomHint, 32, sizeof(MotifWmHints) / sizeof(uint32_t), &hint);
            } else {
                xcb.pxcb_delete_property(handle.getNativeHandle(), window, atomHint);
            }
        }
        
        inline void setNetMWState(XCBHandle& handle, xcb_window_t rootWindow, 
            xcb_window_t window, 
            bool set, 
            xcb_atom_t first, 
            xcb_atom_t second = XCB_NONE) {
            auto& xcb = handle.getXcbFunctionTable();

            xcb_client_message_event_t event;
            event.response_type = XCB_CLIENT_MESSAGE;
            event.type = handle._NET_WM_STATE.fetch();
            event.window = window;
            event.format = 32;
            event.sequence = 0;
            event.data.data32[0] = set ? 1l : 0l;
            event.data.data32[1] = first;
            event.data.data32[2] = second;
            event.data.data32[3] = 1;
            event.data.data32[4] = 0;
            xcb.pxcb_send_event(handle, 0, rootWindow, 
                XCB_EVENT_MASK_STRUCTURE_NOTIFY | XCB_EVENT_MASK_SUBSTRUCTURE_REDIRECT, reinterpret_cast<const char*>(&event));
        }

        inline  const xcb_screen_t* primaryScreen(XCBHandle& handle) {
            auto& xcb = handle.getXcbFunctionTable();
            const xcb_setup_t *setup = xcb.pxcb_get_setup(handle);
            xcb_screen_t *screen = xcb.pxcb_setup_roots_iterator(setup).data;
            return screen;
        }
}


#endif
#endif // C_XCB_HANDLER_XCB
