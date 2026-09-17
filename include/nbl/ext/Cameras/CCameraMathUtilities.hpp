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

template<typename T>
struct SRigidTransformComponents
{
    hlsl::vector<T, 3> translation = hlsl::vector<T, 3>(T(0));
    hlsl::math::quaternion<T> orientation = hlsl::math::quaternion<T>::identity();
    hlsl::vector<T, 3> scale = hlsl::vector<T, 3>(T(1));
};

template<typename T>
struct SCameraPoseDelta
{
    T position = T(0);
    T rotationDeg = T(0);
};

struct SCameraViewRigDefaults final
{
    static constexpr hlsl::float64_t DegreesToRadians = hlsl::numbers::pi<hlsl::float64_t> / 180.0;
    static constexpr hlsl::float64_t FullTurnDeg = 360.0;
    static constexpr hlsl::float64_t RightAngleDeg = FullTurnDeg / 4.0;
    static constexpr hlsl::float64_t ArcballPitchMarginDeg = 1.0;
    static constexpr hlsl::float64_t DollyPitchMarginDeg = 5.0;
    static constexpr hlsl::float64_t FpsVerticalPitchMarginDeg = 2.0;
    // Arcball and turntable pitch stop short of +-90 deg to avoid a singular up axis.
    static constexpr hlsl::float64_t ArcballPitchLimitDeg = RightAngleDeg - ArcballPitchMarginDeg;
    static constexpr hlsl::float64_t TurntablePitchLimitDeg = ArcballPitchLimitDeg;
    // Chase rigs keep a narrower pitch envelope so the followed subject stays readable.
    static constexpr hlsl::float64_t ChaseMaxPitchDeg = 70.0;
    static constexpr hlsl::float64_t ChaseMinPitchDeg = -60.0;
    // Dolly and FPS rigs also stop short of straight up/down.
    static constexpr hlsl::float64_t DollyPitchLimitDeg = RightAngleDeg - DollyPitchMarginDeg;
    static constexpr hlsl::float64_t FpsVerticalPitchLimitDeg = RightAngleDeg - FpsVerticalPitchMarginDeg;
    // TODO: -90 deg is an elevation of -90, which puts the top-down camera BELOW its target looking up
    // (positive elevation is above the target; compare `IsometricPitchRad`, which is positive). The yaw
    // recovery in `CTopDownCamera` is derived for +90. Fix: `TopDownPitchDeg = RightAngleDeg`. Not applied
    // yet because it changes visible behaviour.
    static constexpr hlsl::float64_t TopDownPitchDeg = -RightAngleDeg;
    // Half of a right angle is the canonical isometric azimuth.
    static constexpr hlsl::float64_t IsometricYawDeg = RightAngleDeg / 2.0;
    // tan(theta) = 1 / sqrt(2) for the canonical isometric pitch.
    static constexpr hlsl::float64_t IsometricPitchTangent = 1.0 / hlsl::numbers::sqrt2<hlsl::float64_t>;
    // atan(1 / sqrt(2)) is the canonical isometric pitch used by the fixed rig.
    static inline const hlsl::float64_t IsometricPitchRad = std::atan(IsometricPitchTangent);
    static inline const hlsl::float64_t IsometricPitchDeg = IsometricPitchRad / DegreesToRadians;

    static inline constexpr hlsl::float64_t ArcballPitchLimitRad = ArcballPitchLimitDeg * DegreesToRadians;
    static inline constexpr hlsl::float64_t TurntablePitchLimitRad = TurntablePitchLimitDeg * DegreesToRadians;
    static inline constexpr hlsl::float64_t ChaseMaxPitchRad = ChaseMaxPitchDeg * DegreesToRadians;
    static inline constexpr hlsl::float64_t ChaseMinPitchRad = ChaseMinPitchDeg * DegreesToRadians;
    static inline constexpr hlsl::float64_t DollyPitchLimitRad = DollyPitchLimitDeg * DegreesToRadians;
    static inline constexpr hlsl::float64_t FpsVerticalPitchLimitRad = FpsVerticalPitchLimitDeg * DegreesToRadians;
    static inline constexpr hlsl::float64_t TopDownPitchRad = TopDownPitchDeg * DegreesToRadians;
    static inline constexpr hlsl::float64_t IsometricYawRad = IsometricYawDeg * DegreesToRadians;
};

struct SCameraRigidMathDefaults final
{
    // Treat vectors as effectively parallel once the normalized dot product stays within 1e-2 of 1.
    static constexpr hlsl::float64_t LookAtParallelThreshold = 1.0 - 1e-2;
};

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
    static inline bool isNearlyZeroScalar(const T value, const T epsilon = hlsl::numeric_limits<T>::epsilon)
    {
        return hlsl::approx::absRelEqual<T>(value, T(0), epsilon, T(0));
    }

    template<typename T, uint32_t N>
    static inline bool isNearlyZeroVector(const hlsl::vector<T, N>& value, const T epsilon = hlsl::numeric_limits<T>::epsilon)
    {
        return hlsl::approx::absRelEqual<T>(hlsl::length(value), T(0), epsilon, T(0));
    }

    // `value` is basis local, so x is right and y is up; the forward component is ignored.
    template<typename T>
    static inline bool hasPlanarDeltaXY(const hlsl::vector<T, 3>& value, const T epsilon = hlsl::numeric_limits<T>::epsilon)
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

    template<typename T>
    static inline constexpr T getCameraLookAtParallelThreshold()
    {
        return static_cast<T>(SCameraRigidMathDefaults::LookAtParallelThreshold);
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

    // TODO: candidate for nbl::hlsl (composes Y * X * Z, unlike `quaternion<T>::createFromEulerAnglesXYZ`)
    template<typename T>
    static inline hlsl::math::quaternion<T> makeQuaternionFromEulerRadiansYXZ(const hlsl::vector<T, 3>& eulerRadians)
    {
        const auto pitch = hlsl::math::quaternion<T>::createFromAxisAngle(getCameraWorldRight<T>(), eulerRadians.x);
        const auto yaw = hlsl::math::quaternion<T>::createFromAxisAngle(getCameraWorldUp<T>(), eulerRadians.y);
        const auto roll = hlsl::math::quaternion<T>::createFromAxisAngle(getCameraWorldForward<T>(), eulerRadians.z);
        return hlsl::normalize(yaw * pitch * roll);
    }

    // TODO: candidate for nbl::hlsl (composes Y * X * Z, unlike `quaternion<T>::createFromEulerAnglesXYZ`)
    template<typename T>
    static inline hlsl::math::quaternion<T> makeQuaternionFromEulerDegreesYXZ(const hlsl::vector<T, 3>& eulerDegrees)
    {
        return makeQuaternionFromEulerRadiansYXZ(hlsl::vector<T, 3>(
            hlsl::radians(eulerDegrees.x),
            hlsl::radians(eulerDegrees.y),
            hlsl::radians(eulerDegrees.z)));
    }

    template<typename T>
    static inline hlsl::math::quaternion<T> makeQuaternionFromBasis(
        const hlsl::vector<T, 3>& right,
        const hlsl::vector<T, 3>& up,
        const hlsl::vector<T, 3>& forward)
    {
        const auto canonicalForward = safeNormalizeVec3(forward, getCameraWorldForward<T>());

        auto canonicalRight = right - canonicalForward * hlsl::dot(right, canonicalForward);
        canonicalRight = safeNormalizeVec3(
            canonicalRight,
            safeNormalizeVec3(hlsl::cross(up, canonicalForward), getCameraWorldRight<T>()));

        auto canonicalUp = hlsl::cross(canonicalForward, canonicalRight);
        canonicalUp = safeNormalizeVec3(
            canonicalUp,
            safeNormalizeVec3(up - canonicalForward * hlsl::dot(up, canonicalForward), getCameraWorldUp<T>()));

        canonicalRight = safeNormalizeVec3(hlsl::cross(canonicalUp, canonicalForward), canonicalRight);
        canonicalUp = safeNormalizeVec3(hlsl::cross(canonicalForward, canonicalRight), canonicalUp);

        const SCameraBasis<T> basis = { canonicalRight, canonicalUp, canonicalForward };
        const auto candidate = hlsl::_static_cast<hlsl::math::quaternion<T>>(basis.getRotationMatrix());
        if (!isFiniteQuaternion(candidate))
            return hlsl::math::quaternion<T>::identity();

        return hlsl::normalize(candidate);
    }

    template<typename T>
    static inline bool tryBuildCameraBasisFromForwardUpHint(
        const hlsl::vector<T, 3>& forwardHint,
        const hlsl::vector<T, 3>& upHint,
        hlsl::vector<T, 3>& outRight,
        hlsl::vector<T, 3>& outUp,
        hlsl::vector<T, 3>& outForward)
    {
        // has to be tested before `safeNormalizeVec3` substitutes the fallback, which always passes the checks
        if (!isFiniteVec3(forwardHint) || isNearlyZeroVector(forwardHint))
            return false;

        const auto forward = safeNormalizeVec3(forwardHint, getCameraWorldForward<T>());
        const auto preferredUp = safeNormalizeVec3(upHint, getCameraWorldUp<T>());
        auto right = hlsl::cross(preferredUp, forward);
        if (!isFiniteVec3(right) || isNearlyZeroVector(right))
        {
            const auto fallbackUp = hlsl::abs(forward.y) < getCameraLookAtParallelThreshold<T>() ?
                getCameraWorldUp<T>() :
                getCameraWorldForward<T>();
            right = hlsl::cross(fallbackUp, forward);
            if (!isFiniteVec3(right) || isNearlyZeroVector(right))
                return false;
        }

        right = safeNormalizeVec3(right, getCameraWorldRight<T>());
        auto up = safeNormalizeVec3(hlsl::cross(forward, right), preferredUp);
        right = safeNormalizeVec3(hlsl::cross(up, forward), right);

        const SCameraBasis<T> basis = { right, up, forward };
        if (!hlsl::math::linalg::RuntimeTraits<hlsl::matrix<T, 3, 3> >::create(basis.getRotationMatrix()).orthonormal)
            return false;

        outRight = right;
        outUp = up;
        outForward = forward;
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

    template<typename T>
    static inline T getPlanarRadiusXZ(const hlsl::vector<T, 3>& offset)
    {
        return hlsl::length(hlsl::vector<T, 2>(offset.x, offset.z));
    }

    template<typename T>
    static inline T getPathDistance(const T pathU, const T pathV)
    {
        return hlsl::length(hlsl::vector<T, 2>(pathU, pathV));
    }

    // NOTE: `pathS` is measured from +X towards +Z, a quarter turn away from the orbit azimuth above.
    template<typename T>
    static inline hlsl::vector<T, 3> makePathOffsetFromState(const T pathS, const T pathU, const T pathV)
    {
        return hlsl::vector<T, 3>(hlsl::cos(pathS) * pathU, pathV, hlsl::sin(pathS) * pathU);
    }

    template<typename T>
    static inline bool sanitizePathState(T& pathS, T& pathU, T& pathV, T& pathRoll, const T minU)
    {
        if (!isFiniteScalar(pathS) || !isFiniteScalar(pathU) || !isFiniteScalar(pathV) || !isFiniteScalar(pathRoll))
            return false;

        pathS = wrapAngleRad(pathS);
        pathU = hlsl::max(minU, pathU);
        pathRoll = wrapAngleRad(pathRoll);
        return isFiniteScalar(pathS) &&
            isFiniteScalar(pathU) &&
            isFiniteScalar(pathV) &&
            isFiniteScalar(pathRoll);
    }

    template<typename T>
    static inline bool tryScalePathStateDistance(
        const T desiredDistance,
        const T minU,
        T& pathU,
        T& pathV,
        T* outAppliedDistance = nullptr)
    {
        if (!isFiniteScalar(desiredDistance) ||
            !isFiniteScalar(pathU) ||
            !isFiniteScalar(pathV))
            return false;

        const T currentDistance = getPathDistance(pathU, pathV);
        constexpr T Epsilon = hlsl::numeric_limits<T>::epsilon;
        if (currentDistance > Epsilon)
        {
            const T scale = desiredDistance / currentDistance;
            pathU = hlsl::max(minU, pathU * scale);
            pathV *= scale;
        }
        else
        {
            pathU = hlsl::max(minU, desiredDistance);
            pathV = T(0);
        }

        if (outAppliedDistance)
            *outAppliedDistance = getPathDistance(pathU, pathV);
        return isFiniteScalar(pathU) && isFiniteScalar(pathV);
    }

    template<typename T>
    static inline bool tryBuildPathStateFromPosition(
        const hlsl::vector<T, 3>& targetPosition,
        const hlsl::vector<T, 3>& position,
        const T minRadius,
        T& outS,
        T& outU,
        T& outV)
    {
        const auto offset = position - targetPosition;
        const auto radius = getPlanarRadiusXZ(offset);
        if (!isFiniteScalar(radius) || !isFiniteScalar(offset.y))
            return false;

        outS = wrapAngleRad(hlsl::atan2(offset.z, offset.x));
        outU = hlsl::max(minRadius, radius);
        outV = offset.y;
        return isFiniteScalar(outS) &&
            isFiniteScalar(outU) &&
            isFiniteScalar(outV);
    }

    template<typename T>
    static inline bool tryBuildLookAtOrientation(
        const hlsl::vector<T, 3>& position,
        const hlsl::vector<T, 3>& targetPosition,
        const hlsl::vector<T, 3>& preferredUp,
        hlsl::math::quaternion<T>& outOrientation)
    {
        const auto toTarget = targetPosition - position;
        hlsl::vector<T, 3> right = hlsl::vector<T, 3>(T(0));
        hlsl::vector<T, 3> up = hlsl::vector<T, 3>(T(0));
        hlsl::vector<T, 3> forward = hlsl::vector<T, 3>(T(0));
        if (!tryBuildCameraBasisFromForwardUpHint(toTarget, preferredUp, right, up, forward))
            return false;

        outOrientation = makeQuaternionFromBasis(right, up, forward);
        return true;
    }

    template<typename T>
    static inline bool tryExtractRigidPoseFromTransform(
        const hlsl::matrix<T, 4, 4>& transform,
        hlsl::vector<T, 3>& outTranslation,
        hlsl::math::quaternion<T>& outOrientation)
    {
        SRigidTransformComponents<T> components;
        if (!tryExtractRigidTransformComponents(transform, components))
            return false;

        outTranslation = components.translation;
        outOrientation = components.orientation;
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
        hlsl::float64_t3 right = hlsl::float64_t3(0.0);
        hlsl::float64_t3 up = hlsl::float64_t3(0.0);
        hlsl::float64_t3 forward = hlsl::float64_t3(0.0);
        if (!tryBuildCameraBasisFromForwardUpHint(-spherePosition, upHint, right, up, forward))
            return false;

        outPose.position = orbit.target + spherePosition;
        outPose.orientation = makeQuaternionFromBasis(right, up, forward);
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

    // TODO: candidate for nbl::hlsl
    template<typename T>
    static inline hlsl::vector<T, 2> getPitchYawFromForwardVector(const hlsl::vector<T, 3>& forward)
    {
        const T planarLength = hlsl::length(hlsl::vector<T, 2>(forward.x, forward.z));
        return hlsl::vector<T, 2>(
            hlsl::atan2(planarLength, forward.y) - hlsl::numbers::pi<T> * T(0.5),
            hlsl::atan2(forward.x, forward.z));
    }

    template<typename T>
    static inline hlsl::vector<T, 2> getPitchYawFromOrientation(const hlsl::math::quaternion<T>& orientation)
    {
        return getPitchYawFromForwardVector(getOrientationBasis(orientation).forward);
    }

    /// @brief Path state `(s, u, v, roll)` -> camera pose, by way of the orbit state around `targetPosition`.
    /// Both pointer outputs are optional (may be null).
    static inline bool tryBuildPathPoseFromState(
        const hlsl::float64_t3& targetPosition,
        const hlsl::float64_t pathS,
        const hlsl::float64_t pathU,
        const hlsl::float64_t pathV,
        const hlsl::float64_t pathRoll,
        const hlsl::float64_t minRadius,
        const hlsl::float64_t minDistance,
        const hlsl::float64_t maxDistance,
        hlsl::float64_t3& outPosition,
        hlsl::math::quaternion<hlsl::float64_t>& outOrientation,
        hlsl::float64_t* outAppliedDistance = nullptr,
        hlsl::float64_t2* outOrbitUv = nullptr)
    {
        if (!isFiniteScalar(pathS) ||
            !isFiniteScalar(pathU) ||
            !isFiniteScalar(pathV) ||
            !isFiniteScalar(pathRoll))
            return false;

        const hlsl::float64_t appliedU = hlsl::max(minRadius, pathU);
        const auto offset = makePathOffsetFromState(pathS, appliedU, pathV);

        STargetOrbit orbit = {};
        if (!tryBuildOrbitFromPosition(targetPosition, targetPosition + offset, minDistance, maxDistance, orbit))
            return false;
        SCameraRigPose pose = {};
        if (!tryBuildPoseFromOrbit(orbit, minDistance, maxDistance, pose, &orbit.distance))
            return false;

        outPosition = pose.position;
        outOrientation = pose.orientation;
        if (!isNearlyZeroScalar(pathRoll, hlsl::numeric_limits<hlsl::float64_t>::epsilon))
        {
            const auto basis = getOrientationBasis(outOrientation);
            const hlsl::float64_t rollCos = hlsl::cos(pathRoll);
            const hlsl::float64_t rollSin = hlsl::sin(pathRoll);
            const auto right = basis.right * rollCos + basis.up * rollSin;
            const auto up = basis.up * rollCos - basis.right * rollSin;
            outOrientation = makeQuaternionFromBasis(right, up, basis.forward);
        }

        if (outAppliedDistance)
            *outAppliedDistance = orbit.distance;
        if (outOrbitUv)
            *outOrbitUv = orbit.angles;
        return true;
    }

    template<typename T>
    static inline hlsl::vector<T, 3> projectWorldVectorToLocalBasis(
        const hlsl::vector<T, 3>& worldVector,
        const hlsl::vector<T, 3>& right,
        const hlsl::vector<T, 3>& up,
        const hlsl::vector<T, 3>& forward)
    {
        return hlsl::vector<T, 3>(
            hlsl::dot(worldVector, right),
            hlsl::dot(worldVector, up),
            hlsl::dot(worldVector, forward));
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

    // TODO: candidate for nbl::hlsl (inverse of `quaternion<T>::createFromEulerAnglesXYZ`)
    template<typename T>
    static inline hlsl::vector<T, 3> getQuaternionEulerRadians(const hlsl::math::quaternion<T>& orientation)
    {
        const auto q = hlsl::normalize(orientation);
        const T x = q.data.x;
        const T y = q.data.y;
        const T z = q.data.z;
        const T w = q.data.w;

        const T pitch = hlsl::atan2(
            T(2) * (y * z + w * x),
            w * w - x * x - y * y + z * z);
        const T yaw = hlsl::asin(hlsl::clamp(
            T(-2) * (x * z - w * y),
            T(-1),
            T(1)));
        const T roll = hlsl::atan2(
            T(2) * (x * y + w * z),
            w * w + x * x - y * y - z * z);

        return hlsl::vector<T, 3>(pitch, yaw, roll);
    }

    // TODO: candidate for nbl::hlsl (inverse of `quaternion<T>::createFromEulerAnglesXYZ`)
    template<typename T>
    static inline hlsl::vector<T, 3> getQuaternionEulerDegrees(const hlsl::math::quaternion<T>& orientation)
    {
        const auto eulerRadians = getQuaternionEulerRadians(orientation);
        return hlsl::vector<T, 3>(
            hlsl::degrees(eulerRadians.x),
            hlsl::degrees(eulerRadians.y),
            hlsl::degrees(eulerRadians.z));
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
    static inline bool tryComputePoseDelta(
        const hlsl::vector<T, 3>& lhsPosition,
        const hlsl::math::quaternion<T>& lhsOrientation,
        const hlsl::vector<T, 3>& rhsPosition,
        const hlsl::math::quaternion<T>& rhsOrientation,
        SCameraPoseDelta<T>& outDelta)
    {
        outDelta = {};

        const auto lhsNormalized = hlsl::normalize(lhsOrientation);
        const auto rhsNormalized = hlsl::normalize(rhsOrientation);
        if (!isFiniteVec3(lhsPosition) || !isFiniteVec3(rhsPosition) ||
            !isFiniteQuaternion(lhsNormalized) || !isFiniteQuaternion(rhsNormalized))
        {
            return false;
        }

        outDelta.position = hlsl::length(lhsPosition - rhsPosition);
        outDelta.rotationDeg = getQuaternionAngularDistanceDegrees(lhsNormalized, rhsNormalized);
        return isFiniteScalar(outDelta.position) && isFiniteScalar(outDelta.rotationDeg);
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

    // TODO: candidate for nbl::hlsl (inverse of `makeQuaternionFromEulerRadiansYXZ`)
    template<typename T>
    static inline hlsl::vector<T, 3> getQuaternionEulerRadiansYXZ(const hlsl::math::quaternion<T>& orientation)
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

    // TODO: candidate for nbl::hlsl (inverse of `makeQuaternionFromEulerDegreesYXZ`)
    template<typename T>
    static inline hlsl::vector<T, 3> getQuaternionEulerDegreesYXZ(const hlsl::math::quaternion<T>& orientation)
    {
        const auto eulerRadians = getQuaternionEulerRadiansYXZ(orientation);
        return hlsl::vector<T, 3>(
            hlsl::degrees(eulerRadians.x),
            hlsl::degrees(eulerRadians.y),
            hlsl::degrees(eulerRadians.z));
    }

    template<typename T>
    static inline hlsl::vector<T, 3> getCameraOrientationEulerRadians(const hlsl::math::quaternion<T>& orientation)
    {
        return getQuaternionEulerRadiansYXZ(orientation);
    }

    template<typename T>
    static inline hlsl::vector<T, 3> getCameraOrientationEulerDegrees(const hlsl::math::quaternion<T>& orientation)
    {
        return getQuaternionEulerDegreesYXZ(orientation);
    }

    // TODO: candidate for nbl::hlsl (belongs next to `makeQuaternionFromEulerRadiansYXZ`)
    template<typename T>
    static inline hlsl::vector<T, 3> getOrientationDeltaEulerRadiansYXZ(
        const hlsl::math::quaternion<T>& from,
        const hlsl::math::quaternion<T>& to)
    {
        const auto deltaQuat = hlsl::inverse(from) * hlsl::normalize(to);
        return getQuaternionEulerRadiansYXZ(deltaQuat);
    }

    template<typename T>
    static inline hlsl::vector<T, 3> getWrappedEulerDistanceDegrees(
        const hlsl::vector<T, 3>& a,
        const hlsl::vector<T, 3>& b)
    {
        return hlsl::vector<T, 3>(
            getWrappedAngleDistanceDegrees(a.x, b.x),
            getWrappedAngleDistanceDegrees(a.y, b.y),
            getWrappedAngleDistanceDegrees(a.z, b.z));
    }

    // TODO: candidate for nbl::hlsl (component-wise reduction, `hlsl::max` only takes two arguments)
    template<typename T>
    static inline T getMaxVectorComponent(const hlsl::vector<T, 3>& value)
    {
        return hlsl::max(value.x, hlsl::max(value.y, value.z));
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

    template<typename T>
    static inline bool tryExtractRigidTransformComponents(
        const hlsl::matrix<T, 4, 4>& transform,
        SRigidTransformComponents<T>& outComponents)
    {
        outComponents.translation = hlsl::vector<T, 3>(transform[0].w, transform[1].w, transform[2].w);

        auto right = hlsl::vector<T, 3>(transform[0].x, transform[1].x, transform[2].x);
        auto up = hlsl::vector<T, 3>(transform[0].y, transform[1].y, transform[2].y);
        auto forward = hlsl::vector<T, 3>(transform[0].z, transform[1].z, transform[2].z);

        outComponents.scale = hlsl::vector<T, 3>(hlsl::length(right), hlsl::length(up), hlsl::length(forward));

        if (!isFiniteVec3(outComponents.translation) || !isFiniteVec3(outComponents.scale))
            return false;

        constexpr T Epsilon = hlsl::numeric_limits<T>::epsilon;
        if (outComponents.scale.x <= Epsilon || outComponents.scale.y <= Epsilon || outComponents.scale.z <= Epsilon)
            return false;

        right /= outComponents.scale.x;
        up /= outComponents.scale.y;
        forward /= outComponents.scale.z;

        const SCameraBasis<T> basis = { right, up, forward };
        if (!hlsl::math::linalg::RuntimeTraits<hlsl::matrix<T, 3, 3> >::create(basis.getRotationMatrix()).orthonormal)
            return false;

        outComponents.orientation = makeQuaternionFromBasis(right, up, forward);
        return isFiniteQuaternion(outComponents.orientation);
    }

    template<typename T>
    static inline bool tryBuildRigidFrameFromTransform(
        const hlsl::matrix<T, 4, 4>& transform,
        hlsl::matrix<T, 4, 4>& outFrame,
        hlsl::math::quaternion<T>& outOrientation)
    {
        SRigidTransformComponents<T> components;
        if (!tryExtractRigidTransformComponents(transform, components))
            return false;

        outOrientation = components.orientation;
        outFrame = composeTransformMatrix(components.translation, components.orientation);
        return true;
    }

    template<typename T>
    static inline bool decomposeTransformMatrix(
        const hlsl::matrix<T, 4, 4>& transform,
        hlsl::vector<T, 3>& outTranslation,
        hlsl::vector<T, 3>& outRotationEulerDegrees,
        hlsl::vector<T, 3>& outScale)
    {
        SRigidTransformComponents<T> components;
        if (!tryExtractRigidTransformComponents(transform, components))
            return false;

        outTranslation = components.translation;
        outScale = components.scale;
        outRotationEulerDegrees = getCameraOrientationEulerDegrees(components.orientation);
        return isFiniteVec3(outRotationEulerDegrees);
    }
};

} // namespace nbl::ext::cameras

#endif // _C_CAMERA_MATH_UTILITIES_HPP_
