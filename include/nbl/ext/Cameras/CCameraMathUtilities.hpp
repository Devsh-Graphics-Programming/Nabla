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

    /// @brief Orthonormal camera basis looking along `forwardHint`, with up as close to `upHint` as the forward allows
    /// and right the physical right of the frame's handedness.
    ///
    /// When the hint is zero or parallel to forward, whichever of the frame's up and forward is further from forward
    /// stands in for it. Fails only for a zero or non-finite `forwardHint`.
    template<typename T>
    static inline bool tryBuildBasis(
        const hlsl::vector<T, 3>& forwardHint,
        const hlsl::vector<T, 3>& upHint,
        const SWorldFrame& frame,
        SCameraBasis<T>& outBasis)
    {
        constexpr T Epsilon = hlsl::numeric_limits<T>::epsilon;
        // has to be tested before `safeNormalizeVec3` substitutes the fallback, which always passes the checks
        if (!isFiniteVec3(forwardHint) || isNearlyZeroVector(forwardHint, Epsilon))
            return false;

        const hlsl::vector<T, 3> frameUp(frame.up);
        const hlsl::vector<T, 3> frameForward(frame.forward);
        const hlsl::vector<T, 3> frameRight(frame.right);

        const T sign = handednessSign<T>(frame.handedness);
        const auto forward = safeNormalizeVec3(forwardHint, frameForward);
        auto right = hlsl::cross(safeNormalizeVec3(upHint, frameUp), forward) * sign;
        if (!isFiniteVec3(right) || isNearlyZeroVector(right, Epsilon))
        {
            // the axis with the smaller component along forward, so the cross product below is at least 1/sqrt(2) long
            const auto fallbackUp = hlsl::abs(hlsl::dot(forward, frameUp)) <= hlsl::abs(hlsl::dot(forward, frameForward)) ? frameUp : frameForward;
            right = hlsl::cross(fallbackUp, forward) * sign;
        }
        right = safeNormalizeVec3(right, frameRight);

        outBasis.right = right;
        outBasis.up = hlsl::cross(forward, right) * sign;
        outBasis.forward = forward;
        return true;
    }

    /// @brief The quaternion whose local X, Y, Z are `basis.right`, `basis.up` and `forward` for a left-handed
    /// camera or `-forward` for a right-handed one.
    ///
    /// Not normalized, and NaN when the three columns are not a rotation (an orthonormal basis of the other
    /// handedness is a mirror).
    template<typename T>
    static inline hlsl::math::quaternion<T> orientationFromBasis(const SCameraBasis<T>& basis, const ECameraHandedness handedness)
    {
        // a quaternion can only hold a proper rotation (det +1). `(right, up, forward)` has det +1 in a left-handed
        // world but det -1 in a right-handed one, so there the forward has to be negated to land on local Z; this is
        // also where the right-handed projection looks (-Z of view space)
        const auto localZ = basis.forward * handednessSign<T>(handedness);
        const hlsl::matrix<T, 3, 3> columns(
            hlsl::vector<T, 3>(basis.right.x, basis.up.x, localZ.x),
            hlsl::vector<T, 3>(basis.right.y, basis.up.y, localZ.y),
            hlsl::vector<T, 3>(basis.right.z, basis.up.z, localZ.z));
        return hlsl::math::quaternion<T>::createFromRotationMatrix(columns, true);
    }

    /// @brief The camera's physical axes for `orientation`: right is local X, up is local Y, and forward is local Z
    /// for a left-handed camera and local -Z for a right-handed one. The inverse of `orientationFromBasis`.
    template<typename T>
    static inline SCameraBasis<T> basisFromOrientation(const hlsl::math::quaternion<T>& orientation, const ECameraHandedness handedness)
    {
        const auto normalized = hlsl::normalize(orientation);
        SCameraBasis<T> basis;
        basis.right = normalized.transformVector(hlsl::vector<T, 3>(T(1), T(0), T(0)), true);
        basis.up = normalized.transformVector(hlsl::vector<T, 3>(T(0), T(1), T(0)), true);
        basis.forward = normalized.transformVector(hlsl::vector<T, 3>(T(0), T(0), handednessSign<T>(handedness)), true);
        return basis;
    }

    /// @brief Unit direction at `yaw` and `pitch` measured in the orthonormal frame (`right`, `up`, `forward`): yaw
    /// turns `forward` toward `right`, pitch lifts it toward `up`; radians.
    template<typename T>
    static inline hlsl::vector<T, 3> directionFromYawPitch(
        const hlsl::vector<T, 3>& right,
        const hlsl::vector<T, 3>& up,
        const hlsl::vector<T, 3>& forward,
        const T yaw,
        const T pitch)
    {
        const T cosPitch = hlsl::cos(pitch);
        return forward * (cosPitch * hlsl::cos(yaw)) + right * (cosPitch * hlsl::sin(yaw)) + up * hlsl::sin(pitch);
    }

    /// @brief d(`directionFromYawPitch`)/d(pitch), a unit vector: tangent to the unit sphere at the direction, pointing
    /// toward increasing pitch, and therefore perpendicular to the direction, which makes it a usable up hint for a
    /// camera placed there, including at the poles.
    template<typename T>
    static inline hlsl::vector<T, 3> upHintFromYawPitch(
        const hlsl::vector<T, 3>& right,
        const hlsl::vector<T, 3>& up,
        const hlsl::vector<T, 3>& forward,
        const T yaw,
        const T pitch)
    {
        const T sinPitch = hlsl::sin(pitch);
        return up * hlsl::cos(pitch) - forward * (sinPitch * hlsl::cos(yaw)) - right * (sinPitch * hlsl::sin(yaw));
    }

    /// @brief Inverse of `directionFromYawPitch` for a unit `direction`: `.x` yaw in (-pi, pi], `.y` pitch in
    /// [-pi/2, pi/2]. Yaw is zero (not NaN) when `direction` is along `up`.
    template<typename T>
    static inline hlsl::vector<T, 2> yawPitchFromDirection(
        const hlsl::vector<T, 3>& right,
        const hlsl::vector<T, 3>& up,
        const hlsl::vector<T, 3>& forward,
        const hlsl::vector<T, 3>& direction)
    {
        return hlsl::vector<T, 2>(
            hlsl::atan2(hlsl::dot(direction, right), hlsl::dot(direction, forward)),
            hlsl::asin(hlsl::clamp(hlsl::dot(direction, up), T(-1), T(1))));
    }

    /// @brief `from` rotated by `angle` radians toward `toward` in the plane of the two, for orthonormal `from` and
    /// `toward`: `from * cos + toward * sin`. The rotation of the pair itself is `rotateTowards(toward, -from, angle)`.
    template<typename T>
    static inline hlsl::vector<T, 3> rotateTowards(const hlsl::vector<T, 3>& from, const hlsl::vector<T, 3>& toward, const T angle)
    {
        return from * hlsl::cos(angle) + toward * hlsl::sin(angle);
    }

    /// @brief Orientation of a camera of `frame` at `position` looking at `targetPosition`, with up as close to
    /// `preferredUp` as the view direction allows. Fails when the two positions coincide or are not finite.
    template<typename T>
    static inline bool tryCreateQuaternionFromLookAt(
        const hlsl::vector<T, 3>& position,
        const hlsl::vector<T, 3>& targetPosition,
        const hlsl::vector<T, 3>& preferredUp,
        const SWorldFrame& frame,
        hlsl::math::quaternion<T>& outOrientation)
    {
        SCameraBasis<T> basis;
        if (!tryBuildBasis(targetPosition - position, preferredUp, frame, basis))
            return false;

        const auto orientation = orientationFromBasis(basis, frame.handedness);
        if (!isFiniteQuaternion(orientation))
            return false;

        outOrientation = hlsl::normalize(orientation);
        return true;
    }

    /// @brief Orbit state -> camera pose: the camera sits at `target + direction(angles) * distance` in the frame
    /// (see `STargetOrbit::angles`) and looks at the target, its up along increasing elevation.
    ///
    /// `distance` is clamped to `[minDistance, maxDistance]` before use; the value actually used is reported
    /// through the optional `outAppliedDistance` (pointer = optional, may be null).
    /// TODO: the clamp is rig policy inside a coordinate conversion; candidate to move to the callers.
    static inline bool tryBuildPoseFromOrbit(
        const STargetOrbit& orbit,
        const hlsl::float64_t minDistance,
        const hlsl::float64_t maxDistance,
        const SWorldFrame& frame,
        SCameraRigPose& outPose,
        hlsl::float64_t* outAppliedDistance = nullptr)
    {
        if (!isFiniteScalar(orbit.angles.x) ||
            !isFiniteScalar(orbit.angles.y) ||
            !isFiniteScalar(orbit.distance))
            return false;

        const hlsl::float64_t appliedDistance = hlsl::clamp(orbit.distance, minDistance, maxDistance);
        const auto spherePosition = directionFromYawPitch(frame.right, frame.up, frame.forward, orbit.angles.x, orbit.angles.y) * appliedDistance;
        const auto upHint = upHintFromYawPitch(frame.right, frame.up, frame.forward, orbit.angles.x, orbit.angles.y);
        SCameraBasis<hlsl::float64_t> basis;
        if (!tryBuildBasis(-spherePosition, upHint, frame, basis))
            return false;

        const auto orientation = orientationFromBasis(basis, frame.handedness);
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
        const SWorldFrame& frame,
        STargetOrbit& outOrbit)
    {
        const auto offset = cameraPosition - target;
        const auto distance = hlsl::length(offset);
        if (!isFiniteScalar(distance) || distance <= hlsl::numeric_limits<hlsl::float64_t>::epsilon)
            return false;

        const auto angles = yawPitchFromDirection(frame.right, frame.up, frame.forward, offset / distance);
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

    // TODO: candidate for nbl::hlsl (inverse of `quaternion<T>::createFromYawPitchRoll`, returned as (pitch, yaw, roll))
    // Euler decomposition of the rotation about the world Y, X and Z axes in that order, the inverse of
    // `createFromYawPitchRoll`. No handedness: the angles describe where the quaternion's local +Z and local +X go,
    // so they read as the look direction's heading and pitch only for a camera whose look direction is local +Z.
    template<typename T>
    static inline hlsl::vector<T, 3> getPitchYawRollRadians(const hlsl::math::quaternion<T>& orientation)
    {
        // columns are the local axes in world space
        const auto rotation = hlsl::_static_cast<hlsl::matrix<T, 3, 3>>(hlsl::normalize(orientation));
        const auto localX = hlsl::vector<T, 3>(rotation[0][0], rotation[1][0], rotation[2][0]);
        const auto localY = hlsl::vector<T, 3>(rotation[0][1], rotation[1][1], rotation[2][1]);
        const auto localZ = hlsl::vector<T, 3>(rotation[0][2], rotation[1][2], rotation[2][2]);

        const T yaw = hlsl::atan2(localZ.x, localZ.z);
        const T c2 = hlsl::length(hlsl::vector<T, 2>(localX.y, localY.y));
        const T pitch = hlsl::atan2(-localZ.y, c2);
        const T s1 = hlsl::sin(yaw);
        const T c1 = hlsl::cos(yaw);
        const T roll = hlsl::atan2(
            s1 * localY.z - c1 * localY.x,
            c1 * localX.x - s1 * localX.z);
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

    // Engine layout: the quaternion's local axes in the columns, translation in the last column, so
    // `mul(M, float4(v, 1))` transforms a point.
    template<typename T>
    static inline hlsl::matrix<T, 4, 4> composeTransformMatrix(
        const hlsl::vector<T, 3>& translation,
        const hlsl::math::quaternion<T>& orientation,
        const hlsl::vector<T, 3>& scale = hlsl::vector<T, 3>(T(1)))
    {
        const auto rotation = hlsl::_static_cast<hlsl::matrix<T, 3, 3>>(hlsl::normalize(orientation));

        hlsl::matrix<T, 4, 4> result;
        for (uint32_t row = 0u; row < 3u; ++row)
            result[row] = hlsl::vector<T, 4>(rotation[row][0] * scale.x, rotation[row][1] * scale.y, rotation[row][2] * scale.z, translation[row]);
        result[3] = hlsl::vector<T, 4>(T(0), T(0), T(0), T(1));
        return result;
    }

    /// @brief Position and orientation of a transform in the engine layout (see `composeTransformMatrix`).
    ///
    /// Any positive scale is divided out of the columns and dropped. Fails for zero-length or non-finite columns, for
    /// a shear (the columns are not orthogonal) and for a mirror (negative determinant), none of which a position
    /// and a quaternion can hold.
    template<typename T>
    static inline bool tryExtractPositionAndQuaternionFromTransform(
        const hlsl::matrix<T, 4, 4>& transform,
        hlsl::vector<T, 3>& outPosition,
        hlsl::math::quaternion<T>& outOrientation)
    {
        const auto position = hlsl::vector<T, 3>(transform[0].w, transform[1].w, transform[2].w);

        auto localX = hlsl::vector<T, 3>(transform[0].x, transform[1].x, transform[2].x);
        auto localY = hlsl::vector<T, 3>(transform[0].y, transform[1].y, transform[2].y);
        auto localZ = hlsl::vector<T, 3>(transform[0].z, transform[1].z, transform[2].z);

        const auto scale = hlsl::vector<T, 3>(hlsl::length(localX), hlsl::length(localY), hlsl::length(localZ));

        if (!isFiniteVec3(position) || !isFiniteVec3(scale))
            return false;

        constexpr T Epsilon = hlsl::numeric_limits<T>::epsilon;
        if (scale.x <= Epsilon || scale.y <= Epsilon || scale.z <= Epsilon)
            return false;

        localX /= scale.x;
        localY /= scale.y;
        localZ /= scale.z;

        const hlsl::matrix<T, 3, 3> rotation(
            hlsl::vector<T, 3>(localX.x, localY.x, localZ.x),
            hlsl::vector<T, 3>(localX.y, localY.y, localZ.y),
            hlsl::vector<T, 3>(localX.z, localY.z, localZ.z));
        const auto orientation = hlsl::math::quaternion<T>::createFromRotationMatrix(rotation, true);
        if (!isFiniteQuaternion(orientation))
            return false;

        outPosition = position;
        outOrientation = hlsl::normalize(orientation);
        return true;
    }
};

} // namespace nbl::ext::cameras

#endif // _C_CAMERA_MATH_UTILITIES_HPP_
