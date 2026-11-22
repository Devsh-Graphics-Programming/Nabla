// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_CAMERAS_S_CAMERA_TYPES_HPP_
#define _NBL_EXT_CAMERAS_S_CAMERA_TYPES_HPP_

#include <cmath>
#include <cstdint>
#include <limits>

#include "nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl"
#include "nbl/builtin/hlsl/cpp_compat/matrix.hlsl"
#include "nbl/builtin/hlsl/cpp_compat/vector.hlsl"
#include "nbl/builtin/hlsl/math/quaternions.hlsl"

// Foundational value types shared by the runtime cameras, the gimbal and the math helpers.
// Nothing here knows about `ICamera`, control frames or input processing; keep it that way.
namespace nbl::ext::cameras
{

/// @brief Handedness of the world a camera lives in. It fixes what the camera's quaternion means.
///
/// Local X is the camera's right and local Y its up in both worlds. The look direction is local +Z in a left-handed
/// world and local -Z in a right-handed one, the convention of the LH/RH projection matrices and of
/// `hlsl::math::linalg::lhLookAt` / `rhLookAt`, so one view-matrix formula serves both. Equivalently, the physical
/// right is `up x forward` in a left-handed world and `forward x up` in a right-handed one.
///
/// The handedness matters wherever a look direction meets local Z or a right is derived from an up and a forward:
/// converting between `SCameraBasis` and a quaternion, and building a basis from a look direction (look-at, orbit).
/// Poses, world and view matrices and controls are the same in both worlds. The README section "Frames and
/// handedness" lists every function that takes one.
enum class ECameraHandedness : uint8_t
{
    LeftHanded,
    RightHanded
};

/// @brief +1 left-handed, -1 right-handed: the sign of the look direction on local Z, and of the cross product
/// that turns (up, forward) into the physical right.
template<typename T>
constexpr T handednessSign(const ECameraHandedness handedness)
{
    return handedness == ECameraHandedness::LeftHanded ? T(1) : T(-1);
}

/// @brief The conventions of the world a camera lives in: which way is up, which way a zero angle refers to, and
/// the handedness. Every camera is built for one frame and keeps it for its lifetime.
///
/// `up` is the axis yaw and orbit azimuth turn about, the horizon the FPS rig keeps level, and the pole of the
/// orbit elevation. `forward` is the zero reference of those angles: an FPS camera at yaw 0 looks along it, an
/// orbit camera at azimuth 0 stands on the `forward` side of its target. `right` is the physical right of the
/// handedness when looking along `forward` with `up` overhead, `up x forward` left-handed and `forward x up`
/// right-handed; positive yaw and azimuth turn toward it. The three are orthonormal.
struct SWorldFrame
{
    hlsl::float64_t3 right = hlsl::float64_t3(1.0, 0.0, 0.0);
    hlsl::float64_t3 up = hlsl::float64_t3(0.0, 1.0, 0.0);
    hlsl::float64_t3 forward = hlsl::float64_t3(0.0, 0.0, 1.0);
    ECameraHandedness handedness = ECameraHandedness::LeftHanded;

    /// @brief +Y up, +Z forward, +X right.
    static inline SWorldFrame yUpLeftHanded()
    {
        return { .right = { 1.0, 0.0, 0.0 }, .up = { 0.0, 1.0, 0.0 }, .forward = { 0.0, 0.0, 1.0 }, .handedness = ECameraHandedness::LeftHanded };
    }
    /// @brief +Y up, -Z forward, +X right: the frame of the identity camera of a right-handed world.
    static inline SWorldFrame yUpRightHanded()
    {
        return { .right = { 1.0, 0.0, 0.0 }, .up = { 0.0, 1.0, 0.0 }, .forward = { 0.0, 0.0, -1.0 }, .handedness = ECameraHandedness::RightHanded };
    }
    /// @brief +Z up, +Y forward, +X right.
    static inline SWorldFrame zUpRightHanded()
    {
        return { .right = { 1.0, 0.0, 0.0 }, .up = { 0.0, 0.0, 1.0 }, .forward = { 0.0, 1.0, 0.0 }, .handedness = ECameraHandedness::RightHanded };
    }

    /// @brief Any up and forward: `up` is normalized, `forward` loses its component along up and is normalized,
    /// `right` is derived for `handedness`. Fails when either is zero, not finite, or the two are parallel.
    static inline bool tryMake(const hlsl::float64_t3& up, const hlsl::float64_t3& forward, const ECameraHandedness handedness, SWorldFrame& outFrame)
    {
        constexpr auto Epsilon = std::numeric_limits<hlsl::float64_t>::epsilon();
        const auto upLength = hlsl::length(up);
        if (!std::isfinite(upLength) || upLength <= Epsilon)
            return false;
        const auto unitUp = up / upLength;

        const auto planarForward = forward - unitUp * hlsl::dot(forward, unitUp);
        const auto forwardLength = hlsl::length(planarForward);
        if (!std::isfinite(forwardLength) || forwardLength <= Epsilon)
            return false;

        outFrame.up = unitUp;
        outFrame.forward = planarForward / forwardLength;
        outFrame.right = hlsl::cross(unitUp, outFrame.forward) * handednessSign<hlsl::float64_t>(handedness);
        outFrame.handedness = handedness;
        return true;
    }
};

/// @brief Canonical camera pose consisting of world-space position and orientation.
///
/// This type stores only pose data. Higher-level types add target-relative,
/// dynamic-perspective, or path-specific state around it.
struct SCameraRigPose
{
    /// @brief Camera origin in world space.
    hlsl::float64_t3 position = hlsl::float64_t3(0.0);
    /// @brief Camera orientation in world space expressed as a unit quaternion: local X is the camera's right, local Y
    /// its up, and local Z is its look direction for a left-handed camera and the opposite of it for a right-handed
    /// one (see `ECameraHandedness`).
    hlsl::math::quaternion<hlsl::float64_t> orientation = hlsl::math::quaternion<hlsl::float64_t>::identity();
};

/// @brief A world-space target plus the spherical coordinates of the camera around it.
///
/// This is the whole state of a target-relative rig; `CCameraMathUtilities::tryBuildPoseFromOrbit`
/// derives the camera pose from it. The angles are measured in the camera's `SWorldFrame`.
struct STargetOrbit
{
    /// @brief Smallest distance accepted by target-relative rigs; guards the divisions by the distance.
    static inline constexpr hlsl::float64_t DefaultMinDistance = 0.1;
    /// @brief Interim unbounded default for the largest distance.
    static inline constexpr hlsl::float64_t DefaultMaxDistance = std::numeric_limits<hlsl::float64_t>::infinity();

    /// @brief Orbited point in world space.
    hlsl::float64_t3 target = hlsl::float64_t3(0.0);
    /// @brief `.x` yaw: azimuth about the frame's up, from the frame's forward toward its right, so at 0 the camera
    /// stands on the forward side of the target. `.y` pitch: elevation above the target's horizon plane (+90 deg
    /// puts the camera straight above the target, looking down). Radians.
    ///
    /// TODO: the rigs accumulate into `.x` without ever wrapping it, so a camera spun in one direction for long
    /// enough drifts towards angles with no precision left. Wrap on write (`CCameraMathUtilities::wrapAngleRad`).
    hlsl::float64_t2 angles = hlsl::float64_t2(0.0);
    /// @brief Camera-to-target distance in world units.
    hlsl::float64_t distance = DefaultMinDistance;
};

/// @brief The three physical axes of a camera in world space: its right, its up and the direction it looks along.
///
/// Converted to and from a quaternion by `CCameraMathUtilities::orientationFromBasis` / `basisFromOrientation` for
/// a given `ECameraHandedness`.
template<typename T>
struct SCameraBasis
{
    hlsl::vector<T, 3> right = hlsl::vector<T, 3>(T(1), T(0), T(0));
    hlsl::vector<T, 3> up = hlsl::vector<T, 3>(T(0), T(1), T(0));
    hlsl::vector<T, 3> forward = hlsl::vector<T, 3>(T(0), T(0), T(1));
};

} // namespace nbl::ext::cameras

#endif // _NBL_EXT_CAMERAS_S_CAMERA_TYPES_HPP_
