
#include "nbl/ui/XcbConnection.h"
#include <mutex>
#include <string>
#include <vector>
#include <xcb/xproto.h>
#ifdef _NBL_PLATFORM_LINUX_

#include "nbl/ui/CClipboardManagerXcb.h"

namespace nbl::ui
{
    std::string CClipboardManagerXcb::getClipboardText() {
        {
            std::unique_lock<std::mutex> lk(m_clipboardMutex);
            m_clipboardResponseCV.wait_until(lk, std::chrono::system_clock::now() + std::chrono::seconds(1));
        }
        std::lock_guard lk(m_clipboardMutex);
        std::string response = std::move(m_clipboardResponse);
        m_clipboardResponse = std::string();
        return response;
    }

    xcb_window_t CClipboardManagerXcb::getClipboardWindow() {
        const auto& xcb = m_xcbConnection->getXcbFunctionTable();
        xcb_window_t window = 0;
        auto cookie = xcb.pxcb_get_selection_owner(m_xcbConnection->getRawConnection(), 
            m_xcbConnection->resolveXCBAtom(m_CLIPBOARD));
        if(xcb_get_selection_owner_reply_t* reply =
            xcb.pxcb_get_selection_owner_reply(m_xcbConnection->getRawConnection(), cookie, nullptr)) {
            core::SRAIIBasedExiter exitReply([reply]() -> void { 
                free(reply);
            });
            return reply->owner;
        }
        return 0;

    }
    bool CClipboardManagerXcb::setClipboardText(const std::string_view& data) {
        std::lock_guard lk(m_clipboardMutex);
        m_stagedClipboard.m_data = data;
        m_stagedClipboard.m_formats = {
            m_xcbConnection->resolveXCBAtom(m_formatUTF8_0),
            m_xcbConnection->resolveXCBAtom(m_formatUTF8_1),
            m_xcbConnection->resolveXCBAtom(m_formatUTF8_2),
            m_xcbConnection->resolveXCBAtom(m_formatGTK),
            m_xcbConnection->resolveXCBAtom(m_formatString),
            m_xcbConnection->resolveXCBAtom(m_formatText),
            m_xcbConnection->resolveXCBAtom(m_formatTextPlain)
        };
        return true;
    }

    void CClipboardManagerXcb::process(const IWindowXcb* window, xcb_generic_event_t* event) {
        const auto& xcb = m_xcbConnection->getXcbFunctionTable();
        
        auto TARGETS = m_xcbConnection->resolveXCBAtom(m_TARGETS);

        switch(event->response_type & ~0x80) {
            // XCB_ATOM
            // Somone is requesting the clipboard data
            case XCB_SELECTION_REQUEST: {
                auto* sne = reinterpret_cast<xcb_selection_request_event_t*>(event);
                if(sne->requestor == window->getXcbWindow()) {
                    if(sne->target == TARGETS) {
                            std::vector<xcb_atom_t> targets;
                            {
                                std::lock_guard lk(m_clipboardMutex);
                                for(auto& format : m_stagedClipboard.m_formats) {
                                    targets.push_back(format);
                                }
                            }
                            targets.push_back(m_xcbConnection->resolveXCBAtom(m_TARGETS));
                            xcb.pxcb_change_property(
                                m_xcbConnection->getRawConnection(),
                                XCB_PROP_MODE_REPLACE,
                                sne->requestor,
                                sne->property,
                                XCB_ATOM,
                                8*sizeof(xcb_atom_t),
                                targets.size(),
                                &targets[0]);
                    } else {
                        std::lock_guard lk(m_clipboardMutex);
                        xcb.pxcb_change_property(
                            m_xcbConnection->getRawConnection(),
                            XCB_PROP_MODE_REPLACE,
                            sne->requestor,
                            sne->property,
                            sne->target,
                            8,
                            m_stagedClipboard.m_data.size(),
                            m_stagedClipboard.m_data.data());
                    }
                }

                 // Notify the "requestor" that we've already updated the property.
                xcb_selection_notify_event_t notify;
                notify.response_type = XCB_SELECTION_NOTIFY;
                notify.pad0          = 0;
                notify.sequence      = 0;
                notify.time          = sne->time;
                notify.requestor     = sne->requestor;
                notify.selection     = sne->selection;
                notify.target        = sne->target;
                notify.property      = sne->property;

                xcb.pxcb_send_event(m_xcbConnection->getRawConnection(), false,
                            sne->requestor,
                            XCB_EVENT_MASK_NO_EVENT, // SelectionNotify events go without mask
                            (const char*)&notify);

                xcb.pxcb_flush(m_xcbConnection->getRawConnection());
                break;
            }
            // Someone else has new content in the clipboard, so is
            // notifying us that we should delete our data now.
            case XCB_SELECTION_CLEAR: {
                auto* sne = reinterpret_cast<xcb_selection_clear_event_t*>(event);
                if (sne->selection == m_xcbConnection->resolveXCBAtom(m_CLIPBOARD)) {
                    std::lock_guard<std::mutex> lock(m_clipboardMutex);
                    m_stagedClipboard.m_formats = {};
                    m_stagedClipboard.m_data = std::string();
                }
                break;
            }
            // we've requested the clipboard data, and this is the reply
            case XCB_SELECTION_NOTIFY: {
                auto* sne = reinterpret_cast<xcb_selection_notify_event_t*>(event);
                if(sne->requestor == window->getXcbWindow()){
                    // xcb.pxcb_get_a
                    xcb_atom_t fieldType = XCB_ATOM;
                    if(sne->target != TARGETS) {
                        fieldType = sne->target;
                    }
                    xcb_get_property_cookie_t cookie = xcb.pxcb_get_property(m_xcbConnection->getRawConnection(), true, 
                        sne->requestor,
                        sne->property, 
                        fieldType, 0, 0x1fffffff); // 0x1fffffff = INT32_MAX / 4
                    if(xcb_get_property_reply_t* reply =
                        xcb.pxcb_get_property_reply(m_xcbConnection->getRawConnection(), cookie, nullptr)) {
                        core::SRAIIBasedExiter exitReply([reply]() -> void { 
                            free(reply);
                        });

                        if(reply->type == m_xcbConnection->resolveXCBAtom(m_INCR)) {
                            assert(false); // TODO
                        } else {
                            const auto* src = reinterpret_cast<const char*>(xcb.pxcb_get_property_value(reply));
                            size_t n = xcb.pxcb_get_property_value_length(reply);
                            {
                                std::lock_guard lk(m_clipboardMutex);
                                m_clipboardResponse = std::string(src, n);
                            }
                            m_clipboardResponseCV.notify_one();
                        }
                    }
                }
                break;
            }
        }
    }
}

#endif