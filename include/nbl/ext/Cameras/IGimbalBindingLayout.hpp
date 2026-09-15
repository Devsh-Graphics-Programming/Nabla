#ifndef _NBL_I_GIMBAL_BINDING_LAYOUT_HPP_
#define _NBL_I_GIMBAL_BINDING_LAYOUT_HPP_

#include <functional>
#include <unordered_map>

#include "CVirtualGimbalEvent.hpp"
#include "nbl/ui/KeyCodes.h"

namespace nbl::ui
{

/// @brief Static mapping from external input domains to virtual gimbal events.
///
/// This type stores binding layout only. It does not process runtime input.
/// Each binding chooses both the semantic virtual event type and the gain used
/// to convert raw producer values into `CVirtualGimbalEvent::magnitude`.
struct IGimbalBindingLayout
{
    IGimbalBindingLayout() {}
    virtual ~IGimbalBindingLayout() {}

    using gimbal_event_t = core::CVirtualGimbalEvent;
    using encode_keyboard_code_t = ui::E_KEY_CODE;
    using encode_mouse_code_t = ui::E_MOUSE_CODE;
    using encode_imguizmo_code_t = gimbal_event_t::VirtualEventType;

    enum BindingDomain : uint8_t
    {
        Keyboard,
        Mouse,
        Imguizmo,

        Count
    };

    struct CKeyInfo
    {
        union
        {
            encode_keyboard_code_t keyboardCode;
            encode_mouse_code_t mouseCode;
            encode_imguizmo_code_t imguizmoCode;
        };

        CKeyInfo(encode_keyboard_code_t code) : keyboardCode(code), type(Keyboard) {}
        CKeyInfo(encode_mouse_code_t code) : mouseCode(code), type(Mouse) {}
        CKeyInfo(encode_imguizmo_code_t code) : imguizmoCode(code), type(Imguizmo) {}

        BindingDomain type;
    };

    struct CHashInfo
    {
        static inline constexpr double DefaultMagnitudeScale = 1.0;

        CHashInfo() {}
        CHashInfo(gimbal_event_t::VirtualEventType _type, const double _magnitudeScale = DefaultMagnitudeScale)
            : event({ .type = _type }), magnitudeScale(_magnitudeScale) {}
        ~CHashInfo() = default;

        /// @brief Virtual event emitted by this binding.
        gimbal_event_t event = {};
        /// @brief Per-binding gain applied when raw input is converted into one virtual-event magnitude.
        double magnitudeScale = DefaultMagnitudeScale;
        /// @brief Runtime latch used by held keyboard and mouse-button bindings.
        bool active = false;
    };

    using keyboard_to_virtual_events_t = std::unordered_map<encode_keyboard_code_t, CHashInfo>;
    using mouse_to_virtual_events_t = std::unordered_map<encode_mouse_code_t, CHashInfo>;
    using imguizmo_to_virtual_events_t = std::unordered_map<encode_imguizmo_code_t, CHashInfo>;

    virtual const keyboard_to_virtual_events_t& getKeyboardVirtualEventMap() const = 0;
    virtual const mouse_to_virtual_events_t& getMouseVirtualEventMap() const = 0;
    virtual const imguizmo_to_virtual_events_t& getImguizmoVirtualEventMap() const = 0;

    virtual void updateKeyboardMapping(const std::function<void(keyboard_to_virtual_events_t&)>& mapKeys) = 0;
    virtual void updateMouseMapping(const std::function<void(mouse_to_virtual_events_t&)>& mapKeys) = 0;
    virtual void updateImguizmoMapping(const std::function<void(imguizmo_to_virtual_events_t&)>& mapKeys) = 0;

    inline void copyBindingLayoutFrom(const IGimbalBindingLayout& layout)
    {
        updateKeyboardMapping([&](auto& map) { map = sanitizeMapping(layout.getKeyboardVirtualEventMap()); });
        updateMouseMapping([&](auto& map) { map = sanitizeMapping(layout.getMouseVirtualEventMap()); });
        updateImguizmoMapping([&](auto& map) { map = sanitizeMapping(layout.getImguizmoVirtualEventMap()); });
    }

    inline void copyBindingLayoutTo(IGimbalBindingLayout& layout) const
    {
        layout.updateKeyboardMapping([&](auto& map) { map = sanitizeMapping(getKeyboardVirtualEventMap()); });
        layout.updateMouseMapping([&](auto& map) { map = sanitizeMapping(getMouseVirtualEventMap()); });
        layout.updateImguizmoMapping([&](auto& map) { map = sanitizeMapping(getImguizmoVirtualEventMap()); });
    }

protected:
    template<typename Map>
    inline static Map sanitizeMapping(const Map& source)
    {
        Map result;
        for (const auto& [code, hash] : source)
            result.emplace(code, typename Map::mapped_type(hash.event.type, hash.magnitudeScale));
        return result;
    }
};

class CGimbalBindingLayoutStorage : public IGimbalBindingLayout
{
public:
    /// @brief Mutable storage for active or preset binding layout.
    using IGimbalBindingLayout::IGimbalBindingLayout;

    CGimbalBindingLayoutStorage() {}
    virtual ~CGimbalBindingLayoutStorage() {}

    virtual void updateKeyboardMapping(const std::function<void(keyboard_to_virtual_events_t&)>& mapKeys) override { mapKeys(m_keyboardVirtualEventMap); }
    virtual void updateMouseMapping(const std::function<void(mouse_to_virtual_events_t&)>& mapKeys) override { mapKeys(m_mouseVirtualEventMap); }
    virtual void updateImguizmoMapping(const std::function<void(imguizmo_to_virtual_events_t&)>& mapKeys) override { mapKeys(m_imguizmoVirtualEventMap); }

    virtual const keyboard_to_virtual_events_t& getKeyboardVirtualEventMap() const override { return m_keyboardVirtualEventMap; }
    virtual const mouse_to_virtual_events_t& getMouseVirtualEventMap() const override { return m_mouseVirtualEventMap; }
    virtual const imguizmo_to_virtual_events_t& getImguizmoVirtualEventMap() const override { return m_imguizmoVirtualEventMap; }

    keyboard_to_virtual_events_t m_keyboardVirtualEventMap;
    mouse_to_virtual_events_t m_mouseVirtualEventMap;
    imguizmo_to_virtual_events_t m_imguizmoVirtualEventMap;
};

} // namespace nbl::ui

#endif
