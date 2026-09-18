// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_CAMERAS_S_CAMERA_CONTROLS_HPP_
#define _NBL_EXT_CAMERAS_S_CAMERA_CONTROLS_HPP_

#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <type_traits>

#include "nbl/builtin/hlsl/cpp_compat/vector.hlsl"

// The control frame is the only input `ICamera::manipulate` takes. Values are physical: world units for the
// translation axes and the distance, radians for the rotation axes and the path roll, path-model units for the
// path coordinates. Nothing here knows who filled the frame. The extension ships one filler, the mouse/keyboard
// controller; a gamepad controller, a script, a solver or a test fills the same struct and calls the same
// `manipulate`. Which frame a translation is applied in, what a rotation pivots on and how a path coordinate is
// interpreted is stated by each camera next to its `AcceptedControls`.

namespace nbl::ext::cameras
{

/// @brief One bit per control a camera can be told to apply, dense from bit 0, plus the group masks.
enum ECameraControlAxis : uint32_t
{
    TranslateX = 1u << 0,
    TranslateY = 1u << 1,
    TranslateZ = 1u << 2,
    /// @brief Pitch.
    RotateX = 1u << 3,
    /// @brief Yaw.
    RotateY = 1u << 4,
    /// @brief Roll.
    RotateZ = 1u << 5,
    /// @brief Along the camera-target line, positive moves away from the target.
    Distance = 1u << 6,
    /// @brief Path-model progress coordinate.
    PathS = 1u << 7,
    /// @brief Path-model first lateral coordinate.
    PathU = 1u << 8,
    /// @brief Path-model second lateral coordinate.
    PathV = 1u << 9,
    /// @brief Roll about the path-model forward axis.
    PathRoll = 1u << 10,

    Translate = TranslateX | TranslateY | TranslateZ,
    Rotate = RotateX | RotateY | RotateZ,
    Path = PathS | PathU | PathV | PathRoll,
    AllControls = Translate | Rotate | Distance | Path
};

/// @brief Number of single-bit axes, counted from the all-mask.
static inline constexpr uint32_t CameraControlAxisCount = static_cast<uint32_t>(std::popcount(static_cast<uint32_t>(ECameraControlAxis::AllControls)));
static_assert(static_cast<uint32_t>(ECameraControlAxis::AllControls) == (1u << CameraControlAxisCount) - 1u, "control axes must be dense from bit 0");
static_assert(CameraControlAxisCount <= 8u * sizeof(std::underlying_type_t<ECameraControlAxis>), "control axes do not fit the enum's underlying type");

/// @brief Index of one single-bit axis, for tables with one entry per axis.
constexpr uint32_t cameraControlAxisIndex(const ECameraControlAxis axis)
{
    return static_cast<uint32_t>(std::countr_zero(static_cast<uint32_t>(axis)));
}

/// @brief Single-bit axis at `index`.
constexpr ECameraControlAxis cameraControlAxisFromIndex(const uint32_t index)
{
    return static_cast<ECameraControlAxis>(1u << index);
}

/// @brief Stable name of one single-bit axis, for logs and persistence.
constexpr std::string_view cameraControlAxisName(const ECameraControlAxis axis)
{
    switch (axis)
    {
        case ECameraControlAxis::TranslateX: return "translate.x";
        case ECameraControlAxis::TranslateY: return "translate.y";
        case ECameraControlAxis::TranslateZ: return "translate.z";
        case ECameraControlAxis::RotateX: return "rotate.x";
        case ECameraControlAxis::RotateY: return "rotate.y";
        case ECameraControlAxis::RotateZ: return "rotate.z";
        case ECameraControlAxis::Distance: return "distance";
        case ECameraControlAxis::PathS: return "path.s";
        case ECameraControlAxis::PathU: return "path.u";
        case ECameraControlAxis::PathV: return "path.v";
        case ECameraControlAxis::PathRoll: return "path.roll";
        default: return "unknown";
    }
}

/// @brief Single-bit axis named by `name`, or `0` when no axis has that name.
constexpr ECameraControlAxis stringToCameraControlAxis(const std::string_view name)
{
    for (uint32_t i = 0u; i < CameraControlAxisCount; ++i)
    {
        const auto axis = cameraControlAxisFromIndex(i);
        if (cameraControlAxisName(axis) == name)
            return axis;
    }
    return static_cast<ECameraControlAxis>(0u);
}

/// @brief One frame of physical deltas to apply to a camera.
struct SCameraControls
{
    /// @brief World units. The camera defines the frame.
    hlsl::float64_t3 translate = hlsl::float64_t3(0.0);
    /// @brief Radians: x pitch, y yaw, z roll. The camera defines the pivot.
    hlsl::float64_t3 rotate = hlsl::float64_t3(0.0);
    /// @brief World units along the camera-target line, positive moves away from the target.
    hlsl::float64_t distance = 0.0;
    /// @brief Path-model coordinates and roll, in the units the active path model defines.
    struct SPath
    {
        hlsl::float64_t s = 0.0;
        hlsl::float64_t u = 0.0;
        hlsl::float64_t v = 0.0;
        hlsl::float64_t roll = 0.0;
    } path;

    /// @brief The field behind one single-bit axis.
    inline hlsl::float64_t& axis(const ECameraControlAxis axis)
    {
        switch (axis)
        {
            case ECameraControlAxis::TranslateX: return translate.x;
            case ECameraControlAxis::TranslateY: return translate.y;
            case ECameraControlAxis::TranslateZ: return translate.z;
            case ECameraControlAxis::RotateX: return rotate.x;
            case ECameraControlAxis::RotateY: return rotate.y;
            case ECameraControlAxis::RotateZ: return rotate.z;
            case ECameraControlAxis::Distance: return distance;
            case ECameraControlAxis::PathS: return path.s;
            case ECameraControlAxis::PathU: return path.u;
            case ECameraControlAxis::PathV: return path.v;
            case ECameraControlAxis::PathRoll: return path.roll;
            default:
                // only single-bit axes name a field
                assert(false);
                return translate.x;
        }
    }

    inline const hlsl::float64_t& axis(const ECameraControlAxis axis) const
    {
        return const_cast<SCameraControls*>(this)->axis(axis);
    }

    /// @brief `ECameraControlAxis` mask of every axis whose value is not zero.
    inline uint32_t nonZeroAxes() const
    {
        uint32_t mask = 0u;
        for (uint32_t i = 0u; i < CameraControlAxisCount; ++i)
        {
            const auto bit = cameraControlAxisFromIndex(i);
            if (axis(bit) != 0.0)
                mask |= bit;
        }
        return mask;
    }

    /// @brief Whether every value is finite.
    inline bool isFinite() const
    {
        for (uint32_t i = 0u; i < CameraControlAxisCount; ++i)
        {
            if (!std::isfinite(axis(cameraControlAxisFromIndex(i))))
                return false;
        }
        return true;
    }

    /// @brief Copy with every axis outside `accepted` set to zero.
    inline SCameraControls masked(const uint32_t accepted) const
    {
        SCameraControls result = *this;
        for (uint32_t i = 0u; i < CameraControlAxisCount; ++i)
        {
            const auto bit = cameraControlAxisFromIndex(i);
            if ((accepted & bit) == 0u)
                result.axis(bit) = 0.0;
        }
        return result;
    }
};

} // namespace nbl::ext::cameras

#endif // _NBL_EXT_CAMERAS_S_CAMERA_CONTROLS_HPP_
