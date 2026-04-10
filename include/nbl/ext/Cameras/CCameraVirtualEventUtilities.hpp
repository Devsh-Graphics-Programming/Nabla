#ifndef _C_CAMERA_VIRTUAL_EVENT_UTILITIES_HPP_
#define _C_CAMERA_VIRTUAL_EVENT_UTILITIES_HPP_

#include <array>
#include <span>
#include <vector>

#include "CCameraMathUtilities.hpp"
#include "ICamera.hpp"

namespace nbl::core
{

/// @brief Positive and negative semantic virtual-event pair for one scalar axis.
struct SCameraVirtualEventAxisBinding final
{
    CVirtualGimbalEvent::VirtualEventType positive = CVirtualGimbalEvent::None;
    CVirtualGimbalEvent::VirtualEventType negative = CVirtualGimbalEvent::None;
};

/// @brief Reusable axis-binding presets shared by helpers that synthesize virtual events.
struct SCameraVirtualEventBindings final
{
    static inline constexpr std::array<SCameraVirtualEventAxisBinding, 3u> LocalTranslation = {{
        { CVirtualGimbalEvent::MoveRight, CVirtualGimbalEvent::MoveLeft },
        { CVirtualGimbalEvent::MoveUp, CVirtualGimbalEvent::MoveDown },
        { CVirtualGimbalEvent::MoveForward, CVirtualGimbalEvent::MoveBackward }
    }};
};

/// @brief Shared helpers for building and analyzing `CVirtualGimbalEvent` batches.
///
/// These utilities are reused by goal replay, path-model control translation,
/// scripted tooling, and smoke checks whenever code needs to convert typed deltas
/// into semantic event streams or inspect those streams on the CPU.
struct CCameraVirtualEventUtilities final
{
public:
    /// @brief Append one signed scalar as either the positive or negative event variant.
    static inline void appendSignedVirtualEvent(
        std::vector<CVirtualGimbalEvent>& events,
        const double value,
        const CVirtualGimbalEvent::VirtualEventType positive,
        const CVirtualGimbalEvent::VirtualEventType negative,
        const double tolerance = static_cast<double>(SCameraToolingThresholds::TinyScalarEpsilon))
    {
        if (!hlsl::CCameraMathUtilities::isFiniteScalar(value) || hlsl::CCameraMathUtilities::isNearlyZeroScalar(value, tolerance))
            return;

        auto& ev = events.emplace_back();
        ev.type = (value > 0.0) ? positive : negative;
        ev.magnitude = hlsl::abs(value);
    }

    /// @brief Append one signed scalar after normalizing it by a caller-provided denominator.
    static inline void appendScaledVirtualEvent(
        std::vector<CVirtualGimbalEvent>& events,
        const double value,
        const double denominator,
        const double tolerance,
        const CVirtualGimbalEvent::VirtualEventType positive,
        const CVirtualGimbalEvent::VirtualEventType negative)
    {
        if (!hlsl::CCameraMathUtilities::isFiniteScalar(denominator) || hlsl::CCameraMathUtilities::isNearlyZeroScalar(denominator, static_cast<double>(SCameraToolingThresholds::TinyScalarEpsilon)))
            return;

        appendSignedVirtualEvent(events, value / denominator, positive, negative, tolerance);
    }

    /// @brief Append one angular delta by comparing it against a tolerance expressed in degrees.
    static inline void appendAngularDeltaEvent(
        std::vector<CVirtualGimbalEvent>& events,
        const double deltaRadians,
        const double denominator,
        const double toleranceDeg,
        const CVirtualGimbalEvent::VirtualEventType positive,
        const CVirtualGimbalEvent::VirtualEventType negative)
    {
        if (!hlsl::CCameraMathUtilities::isFiniteScalar(deltaRadians) ||
            hlsl::CCameraMathUtilities::isNearlyZeroScalar(hlsl::degrees(deltaRadians), toleranceDeg))
        {
            return;
        }

        appendScaledVirtualEvent(
            events,
            deltaRadians,
            denominator,
            hlsl::radians(toleranceDeg),
            positive,
            negative);
    }

    /// @brief Append one 3-axis scalar bundle through a caller-provided binding table.
    static inline void appendScaledVirtualAxisEvents(
        std::vector<CVirtualGimbalEvent>& events,
        const hlsl::float64_t3& values,
        const hlsl::float64_t3& denominators,
        const hlsl::float64_t3& tolerances,
        const std::array<SCameraVirtualEventAxisBinding, 3u>& axisBindings)
    {
        for (size_t axisIx = 0u; axisIx < axisBindings.size(); ++axisIx)
        {
            appendScaledVirtualEvent(
                events,
                values[axisIx],
                denominators[axisIx],
                tolerances[axisIx],
                axisBindings[axisIx].positive,
                axisBindings[axisIx].negative);
        }
    }

    /// @brief Append a local-space translation delta as semantic move events.
    static inline void appendLocalTranslationEvents(
        std::vector<CVirtualGimbalEvent>& events,
        const hlsl::float64_t3& localDelta,
        const hlsl::float64_t3& denominators = hlsl::float64_t3(1.0),
        const hlsl::float64_t3& tolerances = hlsl::float64_t3(SCameraToolingThresholds::TinyScalarEpsilon))
    {
        appendScaledVirtualAxisEvents(
            events,
            localDelta,
            denominators,
            tolerances,
            SCameraVirtualEventBindings::LocalTranslation);
    }

    /// @brief Reinterpret a world-space translation delta in the local frame of a camera orientation.
    static inline void appendWorldTranslationAsLocalEvents(
        std::vector<CVirtualGimbalEvent>& events,
        const hlsl::camera_quaternion_t<hlsl::float64_t>& orientation,
        const hlsl::float64_t3& worldDelta,
        const hlsl::float64_t3& denominators = hlsl::float64_t3(1.0),
        const hlsl::float64_t3& tolerances = hlsl::float64_t3(SCameraToolingThresholds::TinyScalarEpsilon))
    {
        appendLocalTranslationEvents(
            events,
            hlsl::CCameraMathUtilities::projectWorldVectorToLocalQuaternionFrame(orientation, worldDelta),
            denominators,
            tolerances);
    }

    /// @brief Append one 3-axis angular delta through a caller-provided binding table.
    static inline void appendAngularAxisEvents(
        std::vector<CVirtualGimbalEvent>& events,
        const hlsl::float64_t3& deltaRadians,
        const hlsl::float64_t3& denominators,
        const hlsl::float64_t3& toleranceDeg,
        const std::array<SCameraVirtualEventAxisBinding, 3u>& axisBindings)
    {
        for (size_t axisIx = 0u; axisIx < axisBindings.size(); ++axisIx)
        {
            appendAngularDeltaEvent(
                events,
                deltaRadians[axisIx],
                denominators[axisIx],
                toleranceDeg[axisIx],
                axisBindings[axisIx].positive,
                axisBindings[axisIx].negative);
        }
    }

    /// @brief Accumulate only translation-related virtual events back into a signed delta vector.
    static inline hlsl::float64_t3 collectSignedTranslationDelta(std::span<const CVirtualGimbalEvent> events)
    {
        hlsl::float64_t3 delta = hlsl::float64_t3(0.0);
        for (const auto& ev : events)
        {
            switch (ev.type)
            {
                case CVirtualGimbalEvent::MoveRight: delta.x += ev.magnitude; break;
                case CVirtualGimbalEvent::MoveLeft: delta.x -= ev.magnitude; break;
                case CVirtualGimbalEvent::MoveUp: delta.y += ev.magnitude; break;
                case CVirtualGimbalEvent::MoveDown: delta.y -= ev.magnitude; break;
                case CVirtualGimbalEvent::MoveForward: delta.z += ev.magnitude; break;
                case CVirtualGimbalEvent::MoveBackward: delta.z -= ev.magnitude; break;
                default: break;
            }
        }
        return delta;
    }
};

} // namespace nbl::core

#endif // _C_CAMERA_VIRTUAL_EVENT_UTILITIES_HPP_

