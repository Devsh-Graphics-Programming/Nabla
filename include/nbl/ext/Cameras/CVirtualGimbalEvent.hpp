#ifndef _NBL_C_VIRTUAL_GIMBAL_EVENT_HPP_
#define _NBL_C_VIRTUAL_GIMBAL_EVENT_HPP_

#include <array>
#include <cstdint>
#include <string_view>

#include "nbl/builtin/hlsl/cpp_compat/vector.hlsl"
#include "nbl/core/math/intutil.h"

namespace nbl::core
{

/// @brief One semantic camera command passed to `ICamera::manipulate(...)`.
///
/// `type` selects the command family. `magnitude` stores the non-negative
/// source-normalized scalar amount for that command. Input binders convert
/// raw keyboard, mouse, scroll, and ImGuizmo data into this representation
/// before the camera sees it. Cameras then convert these virtual magnitudes
/// into camera-local motion through their runtime scales and family-specific
/// legalization rules.
struct CVirtualGimbalEvent
{
    /// @brief Bitmask identifiers for semantic movement, rotation, and scale commands.
    enum VirtualEventType : uint32_t
    {
        None = 0,

        MoveForward = core::createBitmask({ 0 }),
        MoveBackward = core::createBitmask({ 1 }),
        MoveLeft = core::createBitmask({ 2 }),
        MoveRight = core::createBitmask({ 3 }),
        MoveUp = core::createBitmask({ 4 }),
        MoveDown = core::createBitmask({ 5 }),
        TiltUp = core::createBitmask({ 6 }),
        TiltDown = core::createBitmask({ 7 }),
        PanLeft = core::createBitmask({ 8 }),
        PanRight = core::createBitmask({ 9 }),
        RollLeft = core::createBitmask({ 10 }),
        RollRight = core::createBitmask({ 11 }),
        ScaleXInc = core::createBitmask({ 12 }),
        ScaleXDec = core::createBitmask({ 13 }),
        ScaleYInc = core::createBitmask({ 14 }),
        ScaleYDec = core::createBitmask({ 15 }),
        ScaleZInc = core::createBitmask({ 16 }),
        ScaleZDec = core::createBitmask({ 17 }),

        EventsCount = 18,

        Translate = MoveForward | MoveBackward | MoveLeft | MoveRight | MoveUp | MoveDown,
        Rotate = TiltUp | TiltDown | PanLeft | PanRight | RollLeft | RollRight,
        Scale = ScaleXInc | ScaleXDec | ScaleYInc | ScaleYDec | ScaleZInc | ScaleZDec,

        All = Translate | Rotate | Scale
    };

    /// @brief Scalar type used to encode one event magnitude.
    using manipulation_encode_t = hlsl::float64_t;

    /// @brief Semantic event identifier.
    VirtualEventType type = None;
    /// @brief Non-negative scalar amount associated with `type`.
    ///
    /// The value is not a raw device unit. It is the virtual amount emitted by
    /// the active input path after applying binding-local gains.
    manipulation_encode_t magnitude = {};

    /// @brief Convert one event identifier to its stable string form.
    static constexpr std::string_view virtualEventToString(VirtualEventType event)
    {
        switch (event)
        {
            case MoveForward: return "MoveForward";
            case MoveBackward: return "MoveBackward";
            case MoveLeft: return "MoveLeft";
            case MoveRight: return "MoveRight";
            case MoveUp: return "MoveUp";
            case MoveDown: return "MoveDown";
            case TiltUp: return "TiltUp";
            case TiltDown: return "TiltDown";
            case PanLeft: return "PanLeft";
            case PanRight: return "PanRight";
            case RollLeft: return "RollLeft";
            case RollRight: return "RollRight";
            case ScaleXInc: return "ScaleXInc";
            case ScaleXDec: return "ScaleXDec";
            case ScaleYInc: return "ScaleYInc";
            case ScaleYDec: return "ScaleYDec";
            case ScaleZInc: return "ScaleZInc";
            case ScaleZDec: return "ScaleZDec";
            case Translate: return "Translate";
            case Rotate: return "Rotate";
            case Scale: return "Scale";
            case None: return "None";
            default: return "Unknown";
        }
    }

    /// @brief Convert one stable string identifier back to an event identifier.
    static constexpr VirtualEventType stringToVirtualEvent(std::string_view event)
    {
        if (event == "MoveForward") return MoveForward;
        if (event == "MoveBackward") return MoveBackward;
        if (event == "MoveLeft") return MoveLeft;
        if (event == "MoveRight") return MoveRight;
        if (event == "MoveUp") return MoveUp;
        if (event == "MoveDown") return MoveDown;
        if (event == "TiltUp") return TiltUp;
        if (event == "TiltDown") return TiltDown;
        if (event == "PanLeft") return PanLeft;
        if (event == "PanRight") return PanRight;
        if (event == "RollLeft") return RollLeft;
        if (event == "RollRight") return RollRight;
        if (event == "ScaleXInc") return ScaleXInc;
        if (event == "ScaleXDec") return ScaleXDec;
        if (event == "ScaleYInc") return ScaleYInc;
        if (event == "ScaleYDec") return ScaleYDec;
        if (event == "ScaleZInc") return ScaleZInc;
        if (event == "ScaleZDec") return ScaleZDec;
        if (event == "Translate") return Translate;
        if (event == "Rotate") return Rotate;
        if (event == "Scale") return Scale;
        if (event == "None") return None;
        return None;
    }

    /// @brief Return whether `event` belongs to the translation subset.
    static constexpr bool isTranslationEvent(const VirtualEventType event)
    {
        return event != None && (event & Translate) == event;
    }

    /// @brief Return whether `event` belongs to the rotation subset.
    static constexpr bool isRotationEvent(const VirtualEventType event)
    {
        return event != None && (event & Rotate) == event;
    }

    /// @brief Return whether `event` belongs to the scale subset.
    static constexpr bool isScaleEvent(const VirtualEventType event)
    {
        return event != None && (event & Scale) == event;
    }

    /// @brief Table listing every individual event bit in declaration order.
    static inline constexpr auto VirtualEventsTypeTable = []()
    {
        std::array<VirtualEventType, EventsCount> output;

        for (uint16_t i = 0u; i < EventsCount; ++i)
            output[i] = static_cast<VirtualEventType>(core::createBitmask({ i }));

        return output;
    }();
};

} // namespace nbl::core

#endif // _NBL_C_VIRTUAL_GIMBAL_EVENT_HPP_
