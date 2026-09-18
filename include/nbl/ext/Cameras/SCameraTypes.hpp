// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_CAMERAS_S_CAMERA_TYPES_HPP_
#define _NBL_EXT_CAMERAS_S_CAMERA_TYPES_HPP_

#include <limits>

#include "nbl/builtin/hlsl/cpp_compat/matrix.hlsl"
#include "nbl/builtin/hlsl/cpp_compat/vector.hlsl"
#include "nbl/builtin/hlsl/math/quaternions.hlsl"

// Foundational value types shared by the runtime cameras, the gimbal and the math helpers.
// Nothing here knows about `ICamera`, control frames or input processing; keep it that way.
namespace nbl::ext::cameras
{

/// @brief Canonical camera pose consisting of world-space position and orientation.
///
/// This type stores only pose data. Higher-level types add target-relative,
/// dynamic-perspective, or path-specific state around it.
struct SCameraRigPose
{
    /// @brief Camera origin in world space.
    hlsl::float64_t3 position = hlsl::float64_t3(0.0);
    /// @brief Camera orientation in world space expressed as a unit quaternion.
    hlsl::math::quaternion<hlsl::float64_t> orientation = hlsl::math::quaternion<hlsl::float64_t>::identity();
};

/// @brief A world-space target plus the spherical coordinates of the camera around it.
///
/// This is the whole state of a target-relative rig; `CCameraMathUtilities::tryBuildPoseFromOrbit`
/// derives the camera pose from it. The polar axis is +Y, matching the camera-local up axis.
struct STargetOrbit
{
    /// @brief Smallest distance accepted by target-relative rigs; guards the divisions by the distance.
    static inline constexpr hlsl::float64_t DefaultMinDistance = 0.1;
    /// @brief Interim unbounded default for the largest distance.
    static inline constexpr hlsl::float64_t DefaultMaxDistance = std::numeric_limits<hlsl::float64_t>::infinity();

    /// @brief Orbited point in world space.
    hlsl::float64_t3 target = hlsl::float64_t3(0.0);
    /// @brief `.x` yaw: azimuth in the XZ plane from +Z towards +X. `.y` pitch: elevation above the target's XZ plane
    /// (+90 deg puts the camera straight above the target, looking down). Radians.
    ///
    /// TODO: the rigs accumulate into `.x` without ever wrapping it, so a camera spun in one direction for long
    /// enough drifts towards angles with no precision left. Wrap on write (`CCameraMathUtilities::wrapAngleRad`).
    hlsl::float64_t2 angles = hlsl::float64_t2(0.0);
    /// @brief Camera-to-target distance in world units.
    hlsl::float64_t distance = DefaultMinDistance;
};

// Orthonormal basis kept as three named vectors; `getRotationMatrix()` is the only place that commits to a matrix
// layout, and it commits to the engine's: basis vectors in the columns, so `mul(R, local)` gives the world vector.
template<typename T>
struct SCameraBasis
{
    hlsl::vector<T, 3> right = hlsl::vector<T, 3>(T(1), T(0), T(0));
    hlsl::vector<T, 3> up = hlsl::vector<T, 3>(T(0), T(1), T(0));
    hlsl::vector<T, 3> forward = hlsl::vector<T, 3>(T(0), T(0), T(1));

    inline hlsl::matrix<T, 3, 3> getRotationMatrix() const
    {
        return hlsl::matrix<T, 3, 3>(
            hlsl::vector<T, 3>(right.x, up.x, forward.x),
            hlsl::vector<T, 3>(right.y, up.y, forward.y),
            hlsl::vector<T, 3>(right.z, up.z, forward.z));
    }
};

} // namespace nbl::ext::cameras

#endif // _NBL_EXT_CAMERAS_S_CAMERA_TYPES_HPP_
