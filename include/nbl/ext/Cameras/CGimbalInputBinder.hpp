#ifndef _NBL_C_GIMBAL_INPUT_BINDER_HPP_
#define _NBL_C_GIMBAL_INPUT_BINDER_HPP_

#include <vector>

#include "IGimbalInputProcessor.hpp"

namespace nbl::ui
{

/// @brief High-level runtime binder for consumers and viewport glue.
///
/// It owns active runtime mappings and collects one frame of virtual events
/// from raw keyboard, mouse, and ImGuizmo input.
class CGimbalInputBinder final : public IGimbalInputProcessor
{
public:
    using base_t = IGimbalInputProcessor;
    using base_t::base_t;
    using input_keyboard_event_t = base_t::input_keyboard_event_t;
    using input_mouse_event_t = base_t::input_mouse_event_t;
    using input_imguizmo_event_t = base_t::input_imguizmo_event_t;

    struct SCollectedVirtualEvents
    {
        /// @brief Concatenated output buffer plus per-domain counts for diagnostics.
        std::vector<gimbal_event_t> events;
        uint32_t keyboardCount = 0u;
        uint32_t mouseCount = 0u;
        uint32_t imguizmoCount = 0u;

        inline uint32_t totalCount() const
        {
            return keyboardCount + mouseCount + imguizmoCount;
        }
    };

    /// @brief Translate one frame of external keyboard, mouse, and ImGuizmo input into virtual events.
    inline void clearActiveBindings()
    {
        updateKeyboardMapping([](auto& map) { map.clear(); });
        updateMouseMapping([](auto& map) { map.clear(); });
        updateImguizmoMapping([](auto& map) { map.clear(); });
    }

    inline void clearBindingLayout()
    {
        clearActiveBindings();
    }

    inline void copyActiveBindingsFromLayout(const IGimbalBindingLayout& layout)
    {
        updateKeyboardMapping([&](auto& map) { map = sanitizeMapping(layout.getKeyboardVirtualEventMap()); });
        updateMouseMapping([&](auto& map) { map = sanitizeMapping(layout.getMouseVirtualEventMap()); });
        updateImguizmoMapping([&](auto& map) { map = sanitizeMapping(layout.getImguizmoVirtualEventMap()); });
    }

    inline void copyBindingLayoutFrom(const IGimbalBindingLayout& layout)
    {
        copyActiveBindingsFromLayout(layout);
    }

    inline void copyActiveBindingsToLayout(IGimbalBindingLayout& layout) const
    {
        layout.updateKeyboardMapping([&](auto& map) { map = sanitizeMapping(getKeyboardVirtualEventMap()); });
        layout.updateMouseMapping([&](auto& map) { map = sanitizeMapping(getMouseVirtualEventMap()); });
        layout.updateImguizmoMapping([&](auto& map) { map = sanitizeMapping(getImguizmoVirtualEventMap()); });
    }

    inline void copyBindingLayoutTo(IGimbalBindingLayout& layout) const
    {
        copyActiveBindingsToLayout(layout);
    }

    inline SCollectedVirtualEvents collectVirtualEvents(
        const std::chrono::microseconds nextPresentationTimeStamp,
        const SUpdateParameters parameters = {})
    {
        beginInputProcessing(nextPresentationTimeStamp);

        SCollectedVirtualEvents output;
        uint32_t keyboardPotentialCount = 0u;
        uint32_t mousePotentialCount = 0u;
        uint32_t imguizmoPotentialCount = 0u;

        processKeyboard(nullptr, keyboardPotentialCount, {});
        processMouse(nullptr, mousePotentialCount, {});
        processImguizmo(nullptr, imguizmoPotentialCount, {});

        output.events.resize(keyboardPotentialCount + mousePotentialCount + imguizmoPotentialCount);
        auto* dst = output.events.data();

        if (keyboardPotentialCount)
        {
            output.keyboardCount = keyboardPotentialCount;
            processKeyboard(dst, output.keyboardCount, parameters.keyboardEvents);
            dst += output.keyboardCount;
        }

        if (mousePotentialCount)
        {
            output.mouseCount = mousePotentialCount;
            processMouse(dst, output.mouseCount, parameters.mouseEvents);
            dst += output.mouseCount;
        }

        if (imguizmoPotentialCount)
        {
            output.imguizmoCount = imguizmoPotentialCount;
            processImguizmo(dst, output.imguizmoCount, parameters.imguizmoEvents);
        }

        endInputProcessing();
        output.events.resize(output.totalCount());
        return output;
    }

private:
    template<typename Map>
    inline static Map sanitizeMapping(const Map& source)
    {
        Map result;
        for (const auto& [code, hash] : source)
            result.emplace(code, typename Map::mapped_type(hash.event.type, hash.magnitudeScale));
        return result;
    }
};

} // namespace nbl::ui

#endif // _NBL_C_GIMBAL_INPUT_BINDER_HPP_
