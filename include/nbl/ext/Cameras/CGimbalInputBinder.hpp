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

        uint32_t totalCount() const;
    };

    /// @brief Translate one frame of external keyboard, mouse, and ImGuizmo input into virtual events.
    void clearActiveBindings();

    void clearBindingLayout();

    void copyActiveBindingsFromLayout(const IGimbalBindingLayout& layout);

    void copyBindingLayoutFrom(const IGimbalBindingLayout& layout);

    void copyActiveBindingsToLayout(IGimbalBindingLayout& layout) const;

    void copyBindingLayoutTo(IGimbalBindingLayout& layout) const;

    SCollectedVirtualEvents collectVirtualEvents(
        const std::chrono::microseconds nextPresentationTimeStamp,
        const SUpdateParameters parameters = {});

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
