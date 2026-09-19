// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_MATH_UTILITIES_HPP_
#define _C_CAMERA_MATH_UTILITIES_HPP_

#include <cmath>

#include "nbl/builtin/hlsl/approx/abs_rel.hlsl"
#include "nbl/builtin/hlsl/approx/vector.hlsl"
#include "nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl"
#include "nbl/builtin/hlsl/cpp_compat/matrix.hlsl"
#include "nbl/builtin/hlsl/cpp_compat/vector.hlsl"
#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/math/linalg/basic.hlsl"
#include "nbl/builtin/hlsl/math/quaternions.hlsl"
#include "nbl/builtin/hlsl/matrix_utils/matrix_runtime_traits.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"

#include "SCameraTypes.hpp"

namespace nbl::ext::cameras
{

struct CCameraMathUtilities final
{
    // TODO: candidate for nbl::hlsl (needs an `hlsl::fmod` first)
    template<typename T>
    static inline T wrapAngleRad(T angle)
    {
        constexpr T Pi = hlsl::numbers::pi<T>;
        constexpr T TwoPi = Pi * static_cast<T>(2);

        angle = std::fmod(angle + Pi, TwoPi);
        if (angle < static_cast<T>(0))
            angle += TwoPi;
        return angle - Pi;
    }

    // TODO: candidate for nbl::hlsl (needs an `hlsl::fmod` first)
    template<typename T>
    static inline T getWrappedAngleDistanceRadians(const T a, const T b)
    {
        return hlsl::abs(wrapAngleRad(a - b));
    }

    // TODO: candidate for nbl::hlsl (needs an `hlsl::fmod` first)
    template<typename T>
    static inline T getWrappedAngleDistanceDegrees(const T a, const T b)
    {
        constexpr T HalfTurn = static_cast<T>(180);
        constexpr T FullTurn = static_cast<T>(360);

        T angle = std::fmod(a - b + HalfTurn, FullTurn);
        if (angle < static_cast<T>(0))
            angle += FullTurn;
        return hlsl::abs(angle - HalfTurn);
    }

    // TODO: candidate for nbl::hlsl (needs an `hlsl::fmod` first)
    template<typename T>
    static inline T lerpWrappedAngleRad(const T a, const T b, const T alpha)
    {
        return a + wrapAngleRad(b - a) * alpha;
    }

    // TODO: candidate for nbl::hlsl (there is no `hlsl::isfinite`, only `isnan`/`isinf`)
    template<typename T>
    static inline bool isFiniteScalar(const T value)
    {
        return !hlsl::isnan(value) && !hlsl::isinf(value);
    }

    template<typename T>
    static inline bool nearlyEqualScalar(const T a, const T b, const T epsilon)
    {
        return hlsl::approx::absRelEqual<T>(a, b, epsilon, T(0));
    }

    template<typename T>
    static inline bool isNearlyZeroScalar(const T value, const T epsilon)
    {
        return hlsl::approx::absRelEqual<T>(value, T(0), epsilon, T(0));
    }

    template<typename T, uint32_t N>
    static inline bool isNearlyZeroVector(const hlsl::vector<T, N>& value, const T epsilon)
    {
        return hlsl::approx::absRelEqual<T>(hlsl::length(value), T(0), epsilon, T(0));
    }

    // `value` is basis local, so x is right and y is up; the forward component is ignored.
    template<typename T>
    static inline bool hasPlanarDeltaXY(const hlsl::vector<T, 3>& value, const T epsilon)
    {
        return !isNearlyZeroVector(hlsl::vector<T, 2>(value.x, value.y), epsilon);
    }

    template<typename VecA, typename VecB, typename T>
    static inline bool nearlyEqualVec3(const VecA& a, const VecB& b, const T epsilon)
    {
        const hlsl::vector<T, 3> delta(
            static_cast<T>(a.x - b.x),
            static_cast<T>(a.y - b.y),
            static_cast<T>(a.z - b.z));
        return hlsl::approx::absRelEqual<T>(hlsl::length(delta), T(0), epsilon, T(0));
    }

    template<typename T>
    static inline constexpr hlsl::vector<T, 3> getCameraWorldRight()
    {
        return hlsl::vector<T, 3>(T(1), T(0), T(0));
    }

    template<typename T>
    static inline constexpr hlsl::vector<T, 3> getCameraWorldUp()
    {
        return hlsl::vector<T, 3>(T(0), T(1), T(0));
    }

    template<typename T>
    static inline constexpr hlsl::vector<T, 3> getCameraWorldForward()
    {
        return hlsl::vector<T, 3>(T(0), T(0), T(1));
    }

    // TODO: candidate for nbl::hlsl (`quaternion<T>` has no factory taking the four components)
    template<typename T>
    static inline hlsl::math::quaternion<T> makeQuaternionFromComponents(const T x, const T y, const T z, const T w)
    {
        hlsl::math::quaternion<T> output;
        output.data = hlsl::vector<T, 4>(x, y, z, w);
        return output;
    }

    template<typename T>
    static inline bool isFiniteQuaternion(const hlsl::math::quaternion<T>& q)
    {
        return isFiniteScalar(q.data.x) &&
            isFiniteScalar(q.data.y) &&
            isFiniteScalar(q.data.z) &&
            isFiniteScalar(q.data.w);
    }

    template<typename T>
    static inline bool isFiniteVec3(const hlsl::vector<T, 3>& value)
    {
        return isFiniteScalar(value.x) &&
            isFiniteScalar(value.y) &&
            isFiniteScalar(value.z);
    }

    // TODO: candidate for nbl::hlsl
    template<typename T>
    static inline hlsl::vector<T, 3> safeNormalizeVec3(const hlsl::vector<T, 3>& value, const hlsl::vector<T, 3>& fallback)
    {
        const auto len = hlsl::length(value);
        if (!isFiniteScalar(len) || len <= hlsl::numeric_limits<T>::epsilon)
            return fallback;
        return value / len;
    }

    /// @brief Orthonormal camera basis looking along `forwardHint`, with up as close to `upHint` as the forward allows.
    ///
    /// Gram-Schmidt on (forward, up hint) finished with a cross product, the same frame `hlsl::math::linalg::lhLookAt`
    /// builds. When the hint is zero or parallel to forward, whichever of world up and world forward is further from
    /// forward stands in for it. Fails only for a zero or non-finite `forwardHint`.
    template<typename T>
    static inline bool tryBuildCameraBasisFromForwardUpHint(
        const hlsl::vector<T, 3>& forwardHint,
        const hlsl::vector<T, 3>& upHint,
        SCameraBasis<T>& outBasis)
    {
        constexpr T Epsilon = hlsl::numeric_limits<T>::epsilon;
        // has to be tested before `safeNormalizeVec3` substitutes the fallback, which always passes the checks
        if (!isFiniteVec3(forwardHint) || isNearlyZeroVector(forwardHint, Epsilon))
            return false;

        const auto forward = safeNormalizeVec3(forwardHint, getCameraWorldForward<T>());
        auto right = hlsl::cross(safeNormalizeVec3(upHint, getCameraWorldUp<T>()), forward);
        if (!isFiniteVec3(right) || isNearlyZeroVector(right, Epsilon))
        {
            // the axis with the smaller component along forward, so the cross product below is at least 1/sqrt(2) long
            const auto fallbackUp = hlsl::abs(forward.y) <= hlsl::abs(forward.z) ?
                getCameraWorldUp<T>() :
                getCameraWorldForward<T>();
            right = hlsl::cross(fallbackUp, forward);
        }
        right = safeNormalizeVec3(right, getCameraWorldRight<T>());

        outBasis.right = right;
        outBasis.up = hlsl::cross(forward, right);
        outBasis.forward = forward;
        return true;
    }

    // `orbitUv.x` is the azimuth in the XZ plane from +Z towards +X, `orbitUv.y` the elevation above it, so the
    // polar axis is +Y, matching `getCameraWorldUp`.
    template<typename T>
    static inline hlsl::vector<T, 3> makeSphericalOffsetFromOrbit(const hlsl::vector<T, 2>& orbitUv, const T distance)
    {
        return hlsl::vector<T, 3>(
            hlsl::cos(orbitUv.y) * hlsl::sin(orbitUv.x) * distance,
            hlsl::sin(orbitUv.y) * distance,
            hlsl::cos(orbitUv.y) * hlsl::cos(orbitUv.x) * distance);
    }

    // d(makeSphericalOffsetFromOrbit)/d(elevation), normalized: tangent to the orbit sphere and therefore
    // perpendicular to the offset, which is what makes it a usable up hint.
    template<typename T>
    static inline hlsl::vector<T, 3> makeSphericalUpFromOrbit(const hlsl::vector<T, 2>& orbitUv)
    {
        return hlsl::vector<T, 3>(
            -hlsl::sin(orbitUv.y) * hlsl::sin(orbitUv.x),
            hlsl::cos(orbitUv.y),
            -hlsl::sin(orbitUv.y) * hlsl::cos(orbitUv.x));
    }

    /// @brief Orientation of a camera at `position` looking at `targetPosition`, with up as close to `preferredUp` as
    /// the view direction allows. Fails when the two positions coincide or are not finite.
    template<typename T>
    static inline bool tryCreateQuaternionFromLookAt(
        const hlsl::vector<T, 3>& position,
        const hlsl::vector<T, 3>& targetPosition,
        const hlsl::vector<T, 3>& preferredUp,
        hlsl::math::quaternion<T>& outOrientation)
    {
        SCameraBasis<T> basis;
        if (!tryBuildCameraBasisFromForwardUpHint(targetPosition - position, preferredUp, basis))
            return false;

        const auto orientation = hlsl::math::quaternion<T>::createFromRotationMatrix(basis.getRotationMatrix(), true);
        if (!isFiniteQuaternion(orientation))
            return false;

        outOrientation = hlsl::normalize(orientation);
        return true;
    }

    /// @brief Orbit state -> camera pose: the camera sits at `target + offset(angles, distance)` and looks at the target.
    ///
    /// `distance` is clamped to `[minDistance, maxDistance]` before use; the value actually used is reported
    /// through the optional `outAppliedDistance` (pointer = optional, may be null).
    /// TODO: the clamp is rig policy inside a coordinate conversion; candidate to move to the callers.
    static inline bool tryBuildPoseFromOrbit(
        const STargetOrbit& orbit,
        const hlsl::float64_t minDistance,
        const hlsl::float64_t maxDistance,
        SCameraRigPose& outPose,
        hlsl::float64_t* outAppliedDistance = nullptr)
    {
        if (!isFiniteScalar(orbit.angles.x) ||
            !isFiniteScalar(orbit.angles.y) ||
            !isFiniteScalar(orbit.distance))
            return false;

        const hlsl::float64_t appliedDistance = hlsl::clamp(orbit.distance, minDistance, maxDistance);
        const auto spherePosition = makeSphericalOffsetFromOrbit(orbit.angles, appliedDistance);
        const auto upHint = safeNormalizeVec3(makeSphericalUpFromOrbit(orbit.angles), getCameraWorldUp<hlsl::float64_t>());
        SCameraBasis<hlsl::float64_t> basis;
        if (!tryBuildCameraBasisFromForwardUpHint(-spherePosition, upHint, basis))
            return false;

        const auto orientation = hlsl::math::quaternion<hlsl::float64_t>::createFromRotationMatrix(basis.getRotationMatrix(), true);
        if (!isFiniteQuaternion(orientation))
            return false;

        outPose.position = orbit.target + spherePosition;
        outPose.orientation = hlsl::normalize(orientation);
        if (outAppliedDistance)
            *outAppliedDistance = appliedDistance;
        return true;
    }

    /// @brief Camera position -> orbit state around `target` (the inverse of `tryBuildPoseFromOrbit` for the position).
    ///
    /// The angles describe the actual position; the distance is clamped to `[minDistance, maxDistance]`.
    /// Fails when the camera sits on the target.
    /// TODO: the clamp is rig policy inside a coordinate conversion; candidate to move to the callers.
    static inline bool tryBuildOrbitFromPosition(
        const hlsl::float64_t3& target,
        const hlsl::float64_t3& cameraPosition,
        const hlsl::float64_t minDistance,
        const hlsl::float64_t maxDistance,
        STargetOrbit& outOrbit)
    {
        const auto offset = cameraPosition - target;
        const auto distance = hlsl::length(offset);
        if (!isFiniteScalar(distance) || distance <= hlsl::numeric_limits<hlsl::float64_t>::epsilon)
            return false;

        const auto local = offset / distance;
        const auto angles = hlsl::float64_t2(
            hlsl::atan2(local.x, local.z),
            hlsl::asin(hlsl::clamp(local.y, -1.0, 1.0)));
        const auto appliedDistance = hlsl::clamp(distance, minDistance, maxDistance);
        if (!isFiniteScalar(angles.x) || !isFiniteScalar(angles.y) || !isFiniteScalar(appliedDistance))
            return false;

        outOrbit.target = target;
        outOrbit.angles = angles;
        outOrbit.distance = appliedDistance;
        return true;
    }

    template<typename T>
    static inline hlsl::vector<T, 3> transformLocalVectorToWorldBasis(
        const hlsl::vector<T, 3>& localVector,
        const hlsl::vector<T, 3>& right,
        const hlsl::vector<T, 3>& up,
        const hlsl::vector<T, 3>& forward)
    {
        return right * localVector.x + up * localVector.y + forward * localVector.z;
    }

    // TODO: candidate for nbl::hlsl (belongs next to `quaternion<T>::slerp`)
    template<typename T>
    static inline T getQuaternionAngularDistanceRadians(const hlsl::math::quaternion<T>& lhs, const hlsl::math::quaternion<T>& rhs)
    {
        const auto lhsNormalized = hlsl::normalize(lhs);
        const auto rhsNormalized = hlsl::normalize(rhs);
        const T orientationDot = hlsl::clamp(
            static_cast<T>(hlsl::abs(hlsl::dot(lhsNormalized.data, rhsNormalized.data))),
            T(0),
            T(1));
        return T(2) * hlsl::acos(orientationDot);
    }

    // TODO: candidate for nbl::hlsl (belongs next to `quaternion<T>::slerp`)
    template<typename T>
    static inline T getQuaternionAngularDistanceDegrees(const hlsl::math::quaternion<T>& lhs, const hlsl::math::quaternion<T>& rhs)
    {
        return hlsl::degrees(getQuaternionAngularDistanceRadians(lhs, rhs));
    }

    template<typename T>
    static inline hlsl::vector<T, 3> projectWorldVectorToLocalQuaternionFrame(
        const hlsl::math::quaternion<T>& orientation,
        const hlsl::vector<T, 3>& worldVector)
    {
        return hlsl::normalize(hlsl::inverse(orientation)).transformVector(worldVector, true);
    }

    template<typename T>
    static inline SCameraBasis<T> getOrientationBasis(const hlsl::math::quaternion<T>& orientation)
    {
        const auto normalized = hlsl::normalize(orientation);
        SCameraBasis<T> basis;
        basis.right = normalized.transformVector(getCameraWorldRight<T>(), true);
        basis.up = normalized.transformVector(getCameraWorldUp<T>(), true);
        basis.forward = normalized.transformVector(getCameraWorldForward<T>(), true);
        return basis;
    }

    // TODO: candidate for nbl::hlsl (inverse of `quaternion<T>::createFromYawPitchRoll`, returned as (pitch, yaw, roll))
    template<typename T>
    static inline hlsl::vector<T, 3> getPitchYawRollRadians(const hlsl::math::quaternion<T>& orientation)
    {
        const auto basis = getOrientationBasis(orientation);
        const T yaw = hlsl::atan2(basis.forward.x, basis.forward.z);
        const T c2 = hlsl::length(hlsl::vector<T, 2>(basis.right.y, basis.up.y));
        const T pitch = hlsl::atan2(-basis.forward.y, c2);
        const T s1 = hlsl::sin(yaw);
        const T c1 = hlsl::cos(yaw);
        const T roll = hlsl::atan2(
            s1 * basis.up.z - c1 * basis.up.x,
            c1 * basis.right.x - s1 * basis.right.z);
        return hlsl::vector<T, 3>(pitch, yaw, roll);
    }

    // TODO: candidate for nbl::hlsl (inverse of `quaternion<T>::createFromYawPitchRoll`, returned as (pitch, yaw, roll))
    template<typename T>
    static inline hlsl::vector<T, 3> getPitchYawRollDegrees(const hlsl::math::quaternion<T>& orientation)
    {
        const auto eulerRadians = getPitchYawRollRadians(orientation);
        return hlsl::vector<T, 3>(
            hlsl::degrees(eulerRadians.x),
            hlsl::degrees(eulerRadians.y),
            hlsl::degrees(eulerRadians.z));
    }

    // TODO: candidate for nbl::hlsl (belongs next to `quaternion<T>::createFromYawPitchRoll`)
    template<typename T>
    static inline hlsl::vector<T, 3> getPitchYawRollDeltaRadians(
        const hlsl::math::quaternion<T>& from,
        const hlsl::math::quaternion<T>& to)
    {
        const auto deltaQuat = hlsl::inverse(from) * hlsl::normalize(to);
        return getPitchYawRollRadians(deltaQuat);
    }

    // Engine layout: basis vectors in the columns, translation in the last column, so `mul(M, float4(v, 1))`
    // transforms a point.
    template<typename T>
    static inline hlsl::matrix<T, 4, 4> composeTransformMatrix(
        const hlsl::vector<T, 3>& translation,
        const hlsl::math::quaternion<T>& orientation,
        const hlsl::vector<T, 3>& scale = hlsl::vector<T, 3>(T(1)))
    {
        const auto basis = getOrientationBasis(orientation);
        const auto scaledRight = basis.right * scale.x;
        const auto scaledUp = basis.up * scale.y;
        const auto scaledForward = basis.forward * scale.z;

        return hlsl::matrix<T, 4, 4>(
            hlsl::vector<T, 4>(scaledRight.x, scaledUp.x, scaledForward.x, translation.x),
            hlsl::vector<T, 4>(scaledRight.y, scaledUp.y, scaledForward.y, translation.y),
            hlsl::vector<T, 4>(scaledRight.z, scaledUp.z, scaledForward.z, translation.z),
            hlsl::vector<T, 4>(T(0), T(0), T(0), T(1)));
    }

    /// @brief Position and orientation of a transform in the engine layout (see `composeTransformMatrix`).
    ///
    /// Any positive scale is divided out of the basis and dropped. Fails for zero-length or non-finite columns, for a
    /// shear (the basis is not orthogonal) and for a mirror (negative determinant), none of which a position and a
    /// quaternion can hold.
    template<typename T>
    static inline bool tryExtractPositionAndQuaternionFromTransform(
        const hlsl::matrix<T, 4, 4>& transform,
        hlsl::vector<T, 3>& outPosition,
        hlsl::math::quaternion<T>& outOrientation)
    {
        const auto position = hlsl::vector<T, 3>(transform[0].w, transform[1].w, transform[2].w);

        auto right = hlsl::vector<T, 3>(transform[0].x, transform[1].x, transform[2].x);
        auto up = hlsl::vector<T, 3>(transform[0].y, transform[1].y, transform[2].y);
        auto forward = hlsl::vector<T, 3>(transform[0].z, transform[1].z, transform[2].z);

        const auto scale = hlsl::vector<T, 3>(hlsl::length(right), hlsl::length(up), hlsl::length(forward));

        if (!isFiniteVec3(position) || !isFiniteVec3(scale))
            return false;

        constexpr T Epsilon = hlsl::numeric_limits<T>::epsilon;
        if (scale.x <= Epsilon || scale.y <= Epsilon || scale.z <= Epsilon)
            return false;

        right /= scale.x;
        up /= scale.y;
        forward /= scale.z;

        const auto orientation = hlsl::math::quaternion<T>::createFromRotationMatrix(SCameraBasis<T>{ right, up, forward }.getRotationMatrix(), true);
        if (!isFiniteQuaternion(orientation))
            return false;

        outPosition = position;
        outOrientation = hlsl::normalize(orientation);
        return true;
    }
};

} // namespace nbl::ext::cameras

#endif // _C_CAMERA_MATH_UTILITIES_HPP_
