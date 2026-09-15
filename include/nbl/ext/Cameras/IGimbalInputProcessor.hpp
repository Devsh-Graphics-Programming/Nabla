#ifndef _NBL_I_GIMBAL_INPUT_PROCESSOR_HPP_
#define _NBL_I_GIMBAL_INPUT_PROCESSOR_HPP_

#include <algorithm>
#include <array>

#include "nbl/builtin/hlsl/tgmath.hlsl"
#include "nbl/ui/KeyCodes.h"
#include "nbl/ui/SInputEvent.h"

#include "IGimbalBindingLayout.hpp"

namespace nbl::ui
{

/// @brief Runtime processor that turns keyboard, mouse, and ImGuizmo input into virtual events.
///
/// Held keyboard and mouse-button bindings emit `frameDeltaSeconds * magnitudeScale`.
/// Relative mouse movement, mouse scroll, and ImGuizmo deltas emit
/// `abs(rawDelta) * magnitudeScale` per bound axis. The result is written into
/// `CVirtualGimbalEvent::magnitude`.
class IGimbalInputProcessor : public CGimbalBindingLayoutStorage
{
public:
    struct SInputProcessorDefaults final
    {
        /// @brief Largest frame interval, in seconds, accepted from held-input accumulation.
        static inline constexpr double MaxFrameDeltaSeconds = 0.2;
        static inline constexpr float ZeroPivot = 0.0f;
        static inline constexpr float UnitPivot = 1.0f;
    };
    static inline constexpr double MaxFrameDeltaSeconds = SInputProcessorDefaults::MaxFrameDeltaSeconds;
    static inline constexpr float ZeroPivot = SInputProcessorDefaults::ZeroPivot;
    static inline constexpr float UnitPivot = SInputProcessorDefaults::UnitPivot;

    using CGimbalBindingLayoutStorage::CGimbalBindingLayoutStorage;

    IGimbalInputProcessor() = default;
    virtual ~IGimbalInputProcessor() = default;

    /// @brief Keyboard events consumed by the processor.
    using input_keyboard_event_t = ui::SKeyboardEvent;

    /// @brief Mouse events consumed by the processor.
    using input_mouse_event_t = ui::SMouseEvent;

    /// @brief ImGuizmo world-space delta transforms consumed by the processor.
    using input_imguizmo_event_t = hlsl::float32_t4x4;

    void beginInputProcessing(const std::chrono::microseconds nextPresentationTimeStamp);

    void endInputProcessing();

    struct SUpdateParameters
    {
        std::span<const input_keyboard_event_t> keyboardEvents = {};
        std::span<const input_mouse_event_t> mouseEvents = {};
        std::span<const input_imguizmo_event_t> imguizmoEvents = {};
    };

    /// @brief Process combined events from `SUpdateParameters` into virtual manipulation events.
    ///
    /// @note This function combines keyboard, mouse, and ImGuizmo processing.
    /// It delegates the actual work to `processKeyboard`, `processMouse`, and
    /// `processImguizmo`, then accumulates their output and total count.
    ///
    /// @param output Pointer to the destination array for generated gimbal events.
    /// Pass `nullptr` to query only the total event count.
    /// @param count Output total number of generated gimbal events.
    /// @param parameters Individual keyboard, mouse, and ImGuizmo input spans.
    void process(gimbal_event_t* output, uint32_t& count, const SUpdateParameters parameters = {});

    /// @brief Process keyboard events into virtual manipulation events.
    ///
    /// @note This function maps keyboard press and release events into virtual
    /// gimbal manipulation events through the active keyboard bindings.
    /// Held keys contribute elapsed seconds scaled by the binding gain.
    ///
    /// @param output Pointer to the destination array for generated gimbal events.
    /// Pass `nullptr` to query only the total event count.
    /// @param count Output number of generated gimbal events.
    /// @param events Keyboard events to process.
    void processKeyboard(gimbal_event_t* output, uint32_t& count, std::span<const input_keyboard_event_t> events);

    /// @brief Process mouse events into virtual manipulation events.
    ///
    /// @note This function maps mouse clicks, scrolls, and movements into
    /// virtual gimbal manipulation events through the active mouse bindings.
    /// Relative movement and scroll contribute absolute signed deltas scaled by
    /// the matching binding gain.
    ///
    /// @param output Pointer to the destination array for generated gimbal events.
    /// Pass `nullptr` to query only the total event count.
    /// @param count Output number of generated gimbal events.
    /// @param events Mouse events to process.
    void processMouse(gimbal_event_t* output, uint32_t& count, std::span<const input_mouse_event_t> events);

    /// @brief Process ImGuizmo transforms into virtual gimbal events.
    ///
    /// @note This function converts world-space delta transforms authored by
    /// ImGuizmo into translation, rotation, and scale virtual events.
    /// Translation uses world-space delta components. Rotation uses extracted
    /// Euler radians. Scale uses multiplicative components around pivot `1`.
    ///
    /// @param output Pointer to the destination array for generated gimbal events.
    /// Pass `nullptr` to query only the total event count.
    /// @param count Output number of generated gimbal events.
    /// @param events ImGuizmo delta transforms to process.
    void processImguizmo(gimbal_event_t* output, uint32_t& count, std::span<const input_imguizmo_event_t> events);

private:
    template<typename EncodeType, uint32_t N>
    struct SEncodedAxisBindingGroup final
    {
        std::array<EncodeType, N> positive = {};
        std::array<EncodeType, N> negative = {};
    };

    struct SInputProcessorBindingGroups final
    {
        static inline constexpr SEncodedAxisBindingGroup<ui::E_MOUSE_CODE, 2u> MouseScroll = {
            .positive = {
                ui::EMC_VERTICAL_POSITIVE_SCROLL,
                ui::EMC_HORIZONTAL_POSITIVE_SCROLL
            },
            .negative = {
                ui::EMC_VERTICAL_NEGATIVE_SCROLL,
                ui::EMC_HORIZONTAL_NEGATIVE_SCROLL
            }
        };

        static inline constexpr SEncodedAxisBindingGroup<ui::E_MOUSE_CODE, 2u> MouseRelativeMovement = {
            .positive = {
                ui::EMC_RELATIVE_POSITIVE_MOVEMENT_X,
                ui::EMC_RELATIVE_POSITIVE_MOVEMENT_Y
            },
            .negative = {
                ui::EMC_RELATIVE_NEGATIVE_MOVEMENT_X,
                ui::EMC_RELATIVE_NEGATIVE_MOVEMENT_Y
            }
        };

        static inline constexpr SEncodedAxisBindingGroup<gimbal_event_t::VirtualEventType, 3u> ImguizmoTranslation = {
            .positive = {
                gimbal_event_t::MoveRight,
                gimbal_event_t::MoveUp,
                gimbal_event_t::MoveForward
            },
            .negative = {
                gimbal_event_t::MoveLeft,
                gimbal_event_t::MoveDown,
                gimbal_event_t::MoveBackward
            }
        };

        static inline constexpr SEncodedAxisBindingGroup<gimbal_event_t::VirtualEventType, 3u> ImguizmoRotation = {
            .positive = {
                gimbal_event_t::TiltUp,
                gimbal_event_t::PanRight,
                gimbal_event_t::RollRight
            },
            .negative = {
                gimbal_event_t::TiltDown,
                gimbal_event_t::PanLeft,
                gimbal_event_t::RollLeft
            }
        };

        static inline constexpr SEncodedAxisBindingGroup<gimbal_event_t::VirtualEventType, 3u> ImguizmoScale = {
            .positive = {
                gimbal_event_t::ScaleXInc,
                gimbal_event_t::ScaleYInc,
                gimbal_event_t::ScaleZInc
            },
            .negative = {
                gimbal_event_t::ScaleXDec,
                gimbal_event_t::ScaleYDec,
                gimbal_event_t::ScaleZDec
            }
        };
    };

    static double clampFrameDeltaTimeSeconds(
        const std::chrono::microseconds nextPresentationTimeStamp,
        const std::chrono::microseconds lastVirtualUpTimeStamp);

    template<typename Map, typename ConsumeFn>
    void processBindingMap(Map& map, gimbal_event_t* output, uint32_t& count, ConsumeFn&& consume)
    {
        count = 0u;
        const auto mappedVirtualEventsCount = static_cast<uint32_t>(map.size());
        if (!output)
        {
            count = mappedVirtualEventsCount;
            return;
        }
        if (!mappedVirtualEventsCount)
            return;

        preprocess(map);
        consume(map);
        postprocess(map, output, count);
    }

    static bool tryGetMouseButtonCode(
        const ui::E_MOUSE_BUTTON button,
        ui::E_MOUSE_CODE& outCode);

    template<typename Map>
    void updateMouseButtonState(Map& map, const input_mouse_event_t::SClickEvent& clickEvent)
    {
        ui::E_MOUSE_CODE mouseCode = ui::EMC_NONE;
        if (!tryGetMouseButtonCode(clickEvent.mouseButton, mouseCode))
            return;

        if (clickEvent.action == input_mouse_event_t::SClickEvent::EA_PRESSED)
            setBindingActiveState(map, mouseCode, true);
        else if (clickEvent.action == input_mouse_event_t::SClickEvent::EA_RELEASED)
            setBindingActiveState(map, mouseCode, false);
    }

    template<typename Code, typename Map>
    void setBindingActiveState(Map& map, const Code code, const bool active)
    {
        const auto request = map.find(code);
        if (request == map.end())
            return;

        request->second.active = active;
    }

    void preprocess(auto& map)
    {
        for (auto& [key, hash] : map)
        {
            hash.event.magnitude = 0.0f;

            if (hash.active)
                hash.event.magnitude = m_frameDeltaSeconds * hash.magnitudeScale;
        }
    }

    void postprocess(const auto& map, gimbal_event_t* output, uint32_t& count)
    {
        for (const auto& [key, hash] : map)
            if (hash.event.magnitude)
            {
                auto* virtualEvent = output + count;
                virtualEvent->type = hash.event.type;
                virtualEvent->magnitude = hash.event.magnitude;
                ++count;
            }
    }

    template <typename EncodeType, typename Map>
    void requestMagnitudeUpdateWithScalar(float signPivot, float dScalar, EncodeType positive, EncodeType negative, Map& map)
    {
        if (dScalar != signPivot)
        {
            const auto dMagnitude = hlsl::abs(dScalar);
            auto code = (dScalar > signPivot) ? positive : negative;
            auto request = map.find(code);
            if (request != map.end())
                request->second.event.magnitude += dMagnitude * request->second.magnitudeScale;
        }
    }

    template <typename EncodeType, typename Map, uint32_t N>
    void requestMagnitudeUpdateWithSignedComponents(
        float signPivot,
        const hlsl::vector<float, N>& components,
        const std::array<EncodeType, N>& positive,
        const std::array<EncodeType, N>& negative,
        Map& map)
    {
        for (uint32_t i = 0u; i < N; ++i)
            requestMagnitudeUpdateWithScalar(signPivot, components[i], positive[i], negative[i], map);
    }

    template <typename EncodeType, typename Map, uint32_t N>
    void requestMagnitudeUpdateWithSignedComponents(
        float signPivot,
        const hlsl::vector<float, N>& components,
        const SEncodedAxisBindingGroup<EncodeType, N>& bindings,
        Map& map)
    {
        requestMagnitudeUpdateWithSignedComponents(
            signPivot,
            components,
            bindings.positive,
            bindings.negative,
            map);
    }

    double m_frameDeltaSeconds = {};
    std::chrono::microseconds m_nextPresentationTimeStamp = {}, m_lastVirtualUpTimeStamp = {};
};

} // namespace nbl::ui

#endif // _NBL_I_GIMBAL_INPUT_PROCESSOR_HPP_
