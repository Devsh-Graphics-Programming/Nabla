#ifndef _C_CAMERA_MATH_UTILITIES_HPP_
#define _C_CAMERA_MATH_UTILITIES_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include "nbl/builtin/hlsl/cpp_compat/vector.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/matrix_utils/transformation_matrix_utils.hlsl"

namespace nbl::hlsl
{

/// @brief Camera-oriented math aliases and helpers built on top of Nabla `nbl::hlsl` types.
template<typename T, uint32_t N>
using camera_vector_t = vector<T, N>;

template<typename T, uint32_t N, uint32_t M>
using camera_matrix_t = matrix<T, N, M>;

template<typename T>
using camera_quaternion_t = math::quaternion<T>;

template<typename T>
struct SRigidTransformComponents
{
    camera_vector_t<T, 3> translation = camera_vector_t<T, 3>(T(0));
    camera_quaternion_t<T> orientation = camera_quaternion_t<T>::create();
    camera_vector_t<T, 3> scale = camera_vector_t<T, 3>(T(1));
};

template<typename T>
struct SCameraPoseDelta
{
    T position = T(0);
    T rotationDeg = T(0);
};

struct SCameraViewRigDefaults final
{
    static constexpr double DegreesToRadians = numbers::pi<double> / 180.0;
    static constexpr double ArcballPitchLimitDeg = 89.0;
    static constexpr double TurntablePitchLimitDeg = ArcballPitchLimitDeg;
    static constexpr double ChaseMaxPitchDeg = 70.0;
    static constexpr double ChaseMinPitchDeg = -60.0;
    static constexpr double DollyPitchLimitDeg = 85.0;
    static constexpr double FpsVerticalPitchLimitDeg = 88.0;
    static constexpr double TopDownPitchDeg = -90.0;
    static constexpr double IsometricYawDeg = 45.0;
    static constexpr double IsometricPitchDeg = 35.264389682754654;

    static inline constexpr double ArcballPitchLimitRad = ArcballPitchLimitDeg * DegreesToRadians;
    static inline constexpr double TurntablePitchLimitRad = TurntablePitchLimitDeg * DegreesToRadians;
    static inline constexpr double ChaseMaxPitchRad = ChaseMaxPitchDeg * DegreesToRadians;
    static inline constexpr double ChaseMinPitchRad = ChaseMinPitchDeg * DegreesToRadians;
    static inline constexpr double DollyPitchLimitRad = DollyPitchLimitDeg * DegreesToRadians;
    static inline constexpr double FpsVerticalPitchLimitRad = FpsVerticalPitchLimitDeg * DegreesToRadians;
    static inline constexpr double TopDownPitchRad = TopDownPitchDeg * DegreesToRadians;
    static inline constexpr double IsometricYawRad = IsometricYawDeg * DegreesToRadians;
    static inline constexpr double IsometricPitchRad = IsometricPitchDeg * DegreesToRadians;
};

struct SCameraRigidMathDefaults final
{
    static constexpr double LookAtParallelThreshold = 0.99;
};

struct CCameraMathUtilities final
{
    template<typename T>
    static inline T wrapAngleRad(T angle)
    {
        constexpr T Pi = numbers::pi<T>;
        constexpr T TwoPi = Pi * static_cast<T>(2);

        angle = std::fmod(angle + Pi, TwoPi);
        if (angle < static_cast<T>(0))
            angle += TwoPi;
        return angle - Pi;
    }

    template<typename T>
    static inline T getWrappedAngleDistanceRadians(const T a, const T b)
    {
        return hlsl::abs(wrapAngleRad(a - b));
    }

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

    template<typename T>
    static inline T lerpWrappedAngleRad(const T a, const T b, const T alpha)
    {
        return a + wrapAngleRad(b - a) * alpha;
    }

    template<typename T>
    static inline bool isFiniteScalar(const T value)
    {
        return std::isfinite(value);
    }

    template<typename T>
    static inline constexpr T getCameraMathEpsilon()
    {
        return std::numeric_limits<T>::epsilon();
    }

    template<typename T>
    static inline bool nearlyEqualScalar(const T a, const T b, const T epsilon)
    {
        return hlsl::abs(a - b) <= epsilon;
    }

    template<typename T>
    static inline bool isNearlyZeroScalar(const T value, const T epsilon = getCameraMathEpsilon<T>())
    {
        return hlsl::abs(value) <= epsilon;
    }

    template<typename T, uint32_t N>
    static inline bool isNearlyZeroVector(const camera_vector_t<T, N>& value, const T epsilon = getCameraMathEpsilon<T>())
    {
        return length(value) <= epsilon;
    }

    template<typename T>
    static inline bool hasPlanarDeltaXY(const camera_vector_t<T, 3>& value, const T epsilon = std::numeric_limits<T>::epsilon())
    {
        return !isNearlyZeroVector(camera_vector_t<T, 2>(value.x, value.y), epsilon);
    }

    template<typename VecA, typename VecB, typename T>
    static inline bool nearlyEqualVec3(const VecA& a, const VecB& b, const T epsilon)
    {
        const camera_vector_t<T, 3> delta(
            static_cast<T>(a.x - b.x),
            static_cast<T>(a.y - b.y),
            static_cast<T>(a.z - b.z));
        return length(delta) <= epsilon;
    }

    template<typename T>
    static inline constexpr camera_vector_t<T, 3> getCameraWorldRight()
    {
        return camera_vector_t<T, 3>(T(1), T(0), T(0));
    }

    template<typename T>
    static inline constexpr camera_vector_t<T, 3> getCameraWorldUp()
    {
        return camera_vector_t<T, 3>(T(0), T(1), T(0));
    }

    template<typename T>
    static inline constexpr camera_vector_t<T, 3> getCameraWorldForward()
    {
        return camera_vector_t<T, 3>(T(0), T(0), T(1));
    }

    template<typename T>
    static inline constexpr T getCameraLookAtParallelThreshold()
    {
        return static_cast<T>(SCameraRigidMathDefaults::LookAtParallelThreshold);
    }

    template<typename T>
    static inline camera_quaternion_t<T> makeIdentityQuaternion()
    {
        return camera_quaternion_t<T>::create();
    }

    template<typename T>
    static inline camera_quaternion_t<T> makeQuaternionFromComponents(const T x, const T y, const T z, const T w)
    {
        camera_quaternion_t<T> output;
        output.data = camera_vector_t<T, 4>(x, y, z, w);
        return output;
    }

    template<typename T>
    static inline camera_quaternion_t<T> normalizeQuaternion(const camera_quaternion_t<T>& q)
    {
        return normalize(q);
    }

    template<typename T>
    static inline bool isFiniteQuaternion(const camera_quaternion_t<T>& q)
    {
        return isFiniteScalar(q.data.x) &&
            isFiniteScalar(q.data.y) &&
            isFiniteScalar(q.data.z) &&
            isFiniteScalar(q.data.w);
    }

    template<typename T>
    static inline bool isFiniteVec3(const camera_vector_t<T, 3>& value)
    {
        return isFiniteScalar(value.x) &&
            isFiniteScalar(value.y) &&
            isFiniteScalar(value.z);
    }

    template<typename T>
    static inline camera_vector_t<T, 3> safeNormalizeVec3(const camera_vector_t<T, 3>& value, const camera_vector_t<T, 3>& fallback)
    {
        const auto len = length(value);
        if (!isFiniteScalar(len) || len <= getCameraMathEpsilon<T>())
            return fallback;
        return value / len;
    }

    template<typename T>
    static inline camera_quaternion_t<T> makeQuaternionFromAxisAngle(const camera_vector_t<T, 3>& axis, const T radians)
    {
        return camera_quaternion_t<T>::create(axis, radians);
    }

    template<typename T>
    static inline camera_quaternion_t<T> makeQuaternionFromEulerRadians(const camera_vector_t<T, 3>& eulerRadians)
    {
        return camera_quaternion_t<T>::create(eulerRadians.x, eulerRadians.y, eulerRadians.z);
    }

    template<typename T>
    static inline camera_quaternion_t<T> makeQuaternionFromEulerDegrees(const camera_vector_t<T, 3>& eulerDegrees)
    {
        return makeQuaternionFromEulerRadians(camera_vector_t<T, 3>(
            radians(eulerDegrees.x),
            radians(eulerDegrees.y),
            radians(eulerDegrees.z)));
    }

    template<typename T>
    static inline camera_quaternion_t<T> makeQuaternionFromEulerRadiansYXZ(const camera_vector_t<T, 3>& eulerRadians)
    {
        const auto pitch = makeQuaternionFromAxisAngle(getCameraWorldRight<T>(), eulerRadians.x);
        const auto yaw = makeQuaternionFromAxisAngle(getCameraWorldUp<T>(), eulerRadians.y);
        const auto roll = makeQuaternionFromAxisAngle(getCameraWorldForward<T>(), eulerRadians.z);
        return normalizeQuaternion(yaw * pitch * roll);
    }

    template<typename T>
    static inline camera_quaternion_t<T> makeQuaternionFromEulerDegreesYXZ(const camera_vector_t<T, 3>& eulerDegrees)
    {
        return makeQuaternionFromEulerRadiansYXZ(camera_vector_t<T, 3>(
            radians(eulerDegrees.x),
            radians(eulerDegrees.y),
            radians(eulerDegrees.z)));
    }

    template<typename T>
    static inline camera_quaternion_t<T> makeQuaternionFromBasis(
        const camera_vector_t<T, 3>& right,
        const camera_vector_t<T, 3>& up,
        const camera_vector_t<T, 3>& forward)
    {
        const auto canonicalForward = safeNormalizeVec3(forward, getCameraWorldForward<T>());

        auto canonicalRight = right - canonicalForward * dot(right, canonicalForward);
        canonicalRight = safeNormalizeVec3(
            canonicalRight,
            safeNormalizeVec3(cross(up, canonicalForward), getCameraWorldRight<T>()));

        auto canonicalUp = cross(canonicalForward, canonicalRight);
        canonicalUp = safeNormalizeVec3(
            canonicalUp,
            safeNormalizeVec3(up - canonicalForward * dot(up, canonicalForward), getCameraWorldUp<T>()));

        canonicalRight = safeNormalizeVec3(cross(canonicalUp, canonicalForward), canonicalRight);
        canonicalUp = safeNormalizeVec3(cross(canonicalForward, canonicalRight), canonicalUp);

        const camera_matrix_t<T, 3, 3> basis { canonicalRight, canonicalUp, canonicalForward };
        const auto desiredRight = canonicalRight;
        const auto desiredUp = canonicalUp;
        const auto desiredForward = canonicalForward;

        const auto scoreCandidate = [&](const camera_quaternion_t<T>& candidate)
        {
            if (!isFiniteQuaternion(candidate))
                return std::numeric_limits<T>::infinity();

            const auto normalizedCandidate = normalizeQuaternion(candidate);
            const auto rebuiltRight = normalizedCandidate.transformVector(camera_vector_t<T, 3>(T(1), T(0), T(0)), true);
            const auto rebuiltUp = normalizedCandidate.transformVector(camera_vector_t<T, 3>(T(0), T(1), T(0)), true);
            const auto rebuiltForward = normalizedCandidate.transformVector(camera_vector_t<T, 3>(T(0), T(0), T(1)), true);

            const T rightError = length(rebuiltRight - desiredRight);
            const T upError = length(rebuiltUp - desiredUp);
            const T forwardError = length(rebuiltForward - desiredForward);
            return rightError + upError + forwardError;
        };

        const auto quaternionFromMatrixFallback = [&](const camera_matrix_t<T, 3, 3>& m)
        {
            const T m00 = m[0][0];
            const T m11 = m[1][1];
            const T m22 = m[2][2];
            const T trace = m00 + m11 + m22;

            camera_quaternion_t<T> output = makeIdentityQuaternion<T>();
            if (trace > T(0))
            {
                const T scale = hlsl::sqrt(trace + T(1));
                const T invScale = T(0.5) / scale;
                output.data.x = (m[2][1] - m[1][2]) * invScale;
                output.data.y = (m[0][2] - m[2][0]) * invScale;
                output.data.z = (m[1][0] - m[0][1]) * invScale;
                output.data.w = scale * T(0.5);
            }
            else if (m00 >= m11 && m00 >= m22)
            {
                const T scale = hlsl::sqrt(T(1) + m00 - m11 - m22);
                const T invScale = T(0.5) / scale;
                output.data.x = scale * T(0.5);
                output.data.y = (m[0][1] + m[1][0]) * invScale;
                output.data.z = (m[2][0] + m[0][2]) * invScale;
                output.data.w = (m[2][1] - m[1][2]) * invScale;
            }
            else if (m11 >= m22)
            {
                const T scale = hlsl::sqrt(T(1) + m11 - m00 - m22);
                const T invScale = T(0.5) / scale;
                output.data.x = (m[0][1] + m[1][0]) * invScale;
                output.data.y = scale * T(0.5);
                output.data.z = (m[1][2] + m[2][1]) * invScale;
                output.data.w = (m[0][2] - m[2][0]) * invScale;
            }
            else
            {
                const T scale = hlsl::sqrt(T(1) + m22 - m00 - m11);
                const T invScale = T(0.5) / scale;
                output.data.x = (m[2][0] + m[0][2]) * invScale;
                output.data.y = (m[1][2] + m[2][1]) * invScale;
                output.data.z = scale * T(0.5);
                output.data.w = (m[1][0] - m[0][1]) * invScale;
            }
            return normalizeQuaternion(output);
        };

        const camera_matrix_t<T, 3, 3> transposedBasis = hlsl::transpose(basis);
        const camera_quaternion_t<T> candidates[] = {
            camera_quaternion_t<T>::create(basis, true),
            camera_quaternion_t<T>::create(transposedBasis, true),
            quaternionFromMatrixFallback(basis),
            quaternionFromMatrixFallback(transposedBasis)
        };

        camera_quaternion_t<T> bestCandidate = makeIdentityQuaternion<T>();
        T bestScore = std::numeric_limits<T>::infinity();
        bool foundFiniteCandidate = false;
        const auto considerCandidate = [&](const camera_quaternion_t<T>& candidate)
        {
            const T score = scoreCandidate(candidate);
            if (score < bestScore)
            {
                bestScore = score;
                bestCandidate = candidate;
                foundFiniteCandidate = true;
            }
        };

        for (const auto& candidate : candidates)
            considerCandidate(candidate);

        if (!foundFiniteCandidate || !isFiniteQuaternion(bestCandidate))
            return makeIdentityQuaternion<T>();

        return normalizeQuaternion(bestCandidate);
    }

    template<typename T>
    static inline bool tryBuildCameraBasisFromForwardUpHint(
        const camera_vector_t<T, 3>& forwardHint,
        const camera_vector_t<T, 3>& upHint,
        camera_vector_t<T, 3>& outRight,
        camera_vector_t<T, 3>& outUp,
        camera_vector_t<T, 3>& outForward)
    {
        const auto forward = safeNormalizeVec3(forwardHint, getCameraWorldForward<T>());
        if (!isFiniteVec3(forward) || isNearlyZeroVector(forward))
            return false;

        const auto preferredUp = safeNormalizeVec3(upHint, getCameraWorldForward<T>());
        auto right = cross(preferredUp, forward);
        if (!isFiniteVec3(right) || isNearlyZeroVector(right))
        {
            const auto fallbackUp = hlsl::abs(forward.z) < getCameraLookAtParallelThreshold<T>() ?
                getCameraWorldForward<T>() :
                getCameraWorldUp<T>();
            right = cross(fallbackUp, forward);
            if (!isFiniteVec3(right) || isNearlyZeroVector(right))
                return false;
        }

        right = safeNormalizeVec3(right, getCameraWorldRight<T>());
        auto up = safeNormalizeVec3(cross(forward, right), preferredUp);
        right = safeNormalizeVec3(cross(up, forward), right);
        if (!isOrthoBase(right, up, forward))
            return false;

        outRight = right;
        outUp = up;
        outForward = forward;
        return true;
    }

    template<typename T>
    static inline camera_vector_t<T, 3> makeSphericalOffsetFromOrbit(const camera_vector_t<T, 2>& orbitUv, const T distance)
    {
        return camera_vector_t<T, 3>(
            hlsl::cos(orbitUv.y) * hlsl::cos(orbitUv.x) * distance,
            hlsl::cos(orbitUv.y) * hlsl::sin(orbitUv.x) * distance,
            hlsl::sin(orbitUv.y) * distance);
    }

    template<typename T>
    static inline T getPlanarRadiusXZ(const camera_vector_t<T, 3>& offset)
    {
        return length(camera_vector_t<T, 2>(offset.x, offset.z));
    }

    template<typename T>
    static inline T getPathDistance(const T pathU, const T pathV)
    {
        return length(camera_vector_t<T, 2>(pathU, pathV));
    }

    template<typename T>
    static inline camera_vector_t<T, 3> makePathOffsetFromState(const T pathS, const T pathU, const T pathV)
    {
        return camera_vector_t<T, 3>(hlsl::cos(pathS) * pathU, pathV, hlsl::sin(pathS) * pathU);
    }

    template<typename T>
    static inline bool sanitizePathState(T& pathS, T& pathU, T& pathV, T& pathRoll, const T minU)
    {
        if (!isFiniteScalar(pathS) || !isFiniteScalar(pathU) || !isFiniteScalar(pathV) || !isFiniteScalar(pathRoll))
            return false;

        pathS = wrapAngleRad(pathS);
        pathU = std::max(minU, pathU);
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
        constexpr T Epsilon = std::numeric_limits<T>::epsilon();
        if (currentDistance > Epsilon)
        {
            const T scale = desiredDistance / currentDistance;
            pathU = std::max(minU, pathU * scale);
            pathV *= scale;
        }
        else
        {
            pathU = std::max(minU, desiredDistance);
            pathV = T(0);
        }

        if (outAppliedDistance)
            *outAppliedDistance = getPathDistance(pathU, pathV);
        return isFiniteScalar(pathU) && isFiniteScalar(pathV);
    }

    template<typename T>
    static inline bool tryBuildPathStateFromPosition(
        const camera_vector_t<T, 3>& targetPosition,
        const camera_vector_t<T, 3>& position,
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
        outU = std::max(minRadius, radius);
        outV = offset.y;
        return isFiniteScalar(outS) &&
            isFiniteScalar(outU) &&
            isFiniteScalar(outV);
    }

    template<typename T>
    static inline bool tryBuildLookAtOrientation(
        const camera_vector_t<T, 3>& position,
        const camera_vector_t<T, 3>& targetPosition,
        const camera_vector_t<T, 3>& preferredUp,
        camera_quaternion_t<T>& outOrientation)
    {
        const auto toTarget = targetPosition - position;
        camera_vector_t<T, 3> right = camera_vector_t<T, 3>(T(0));
        camera_vector_t<T, 3> up = camera_vector_t<T, 3>(T(0));
        camera_vector_t<T, 3> forward = camera_vector_t<T, 3>(T(0));
        if (!tryBuildCameraBasisFromForwardUpHint(toTarget, preferredUp, right, up, forward))
            return false;

        outOrientation = makeQuaternionFromBasis(right, up, forward);
        return true;
    }

    template<typename T>
    static inline bool tryExtractRigidPoseFromTransform(
        const camera_matrix_t<T, 4, 4>& transform,
        camera_vector_t<T, 3>& outTranslation,
        camera_quaternion_t<T>& outOrientation)
    {
        SRigidTransformComponents<T> components;
        if (!tryExtractRigidTransformComponents(transform, components))
            return false;

        outTranslation = components.translation;
        outOrientation = components.orientation;
        return true;
    }

    template<typename T>
    static inline bool tryBuildSphericalPoseFromOrbit(
        const camera_vector_t<T, 3>& targetPosition,
        const camera_vector_t<T, 2>& orbitUv,
        const T distance,
        const T minDistance,
        const T maxDistance,
        camera_vector_t<T, 3>& outPosition,
        camera_quaternion_t<T>& outOrientation,
        T* outAppliedDistance = nullptr)
    {
        if (!isFiniteScalar(orbitUv.x) ||
            !isFiniteScalar(orbitUv.y) ||
            !isFiniteScalar(distance))
            return false;

        const T appliedDistance = std::clamp(distance, minDistance, maxDistance);
        const auto spherePosition = makeSphericalOffsetFromOrbit(orbitUv, appliedDistance);
        const auto upHint = safeNormalizeVec3(
            camera_vector_t<T, 3>(
                -hlsl::sin(orbitUv.y) * hlsl::cos(orbitUv.x),
                -hlsl::sin(orbitUv.y) * hlsl::sin(orbitUv.x),
                hlsl::cos(orbitUv.y)),
            getCameraWorldForward<T>());
        camera_vector_t<T, 3> right = camera_vector_t<T, 3>(T(0));
        camera_vector_t<T, 3> up = camera_vector_t<T, 3>(T(0));
        camera_vector_t<T, 3> forward = camera_vector_t<T, 3>(T(0));
        if (!tryBuildCameraBasisFromForwardUpHint(-spherePosition, upHint, right, up, forward))
            return false;

        outPosition = targetPosition + spherePosition;
        outOrientation = makeQuaternionFromBasis(right, up, forward);
        if (outAppliedDistance)
            *outAppliedDistance = appliedDistance;
        return true;
    }

    template<typename T>
    static inline bool tryBuildOrbitFromPosition(
        const camera_vector_t<T, 3>& targetPosition,
        const camera_vector_t<T, 3>& position,
        const T minDistance,
        const T maxDistance,
        camera_vector_t<T, 2>& outOrbitUv,
        T& outDistance)
    {
        const auto offset = position - targetPosition;
        const auto distance = length(offset);
        if (!isFiniteScalar(distance) || distance <= getCameraMathEpsilon<T>())
            return false;

        outDistance = std::clamp(distance, minDistance, maxDistance);
        const auto local = offset / outDistance;
        outOrbitUv = camera_vector_t<T, 2>(
            hlsl::atan2(local.y, local.x),
            hlsl::asin(std::clamp(local.z, T(-1), T(1))));
        return isFiniteScalar(outOrbitUv.x) &&
            isFiniteScalar(outOrbitUv.y) &&
            isFiniteScalar(outDistance);
    }

    template<typename T>
    static inline camera_vector_t<T, 2> getPitchYawFromForwardVector(const camera_vector_t<T, 3>& forward)
    {
        const T planarLength = length(camera_vector_t<T, 2>(forward.x, forward.z));
        return camera_vector_t<T, 2>(
            hlsl::atan2(planarLength, forward.y) - numbers::pi<T> * T(0.5),
            hlsl::atan2(forward.x, forward.z));
    }

    template<typename T>
    static inline camera_vector_t<T, 2> getPitchYawFromOrientation(const camera_quaternion_t<T>& orientation)
    {
        const auto forward = normalizeQuaternion(orientation).transformVector(camera_vector_t<T, 3>(T(0), T(0), T(1)), true);
        return getPitchYawFromForwardVector(forward);
    }

    template<typename T>
    static inline bool tryBuildPathPoseFromState(
        const camera_vector_t<T, 3>& targetPosition,
        const T pathS,
        const T pathU,
        const T pathV,
        const T pathRoll,
        const T minRadius,
        const T minDistance,
        const T maxDistance,
        camera_vector_t<T, 3>& outPosition,
        camera_quaternion_t<T>& outOrientation,
        T* outAppliedDistance = nullptr,
        camera_vector_t<T, 2>* outOrbitUv = nullptr)
    {
        if (!isFiniteScalar(pathS) ||
            !isFiniteScalar(pathU) ||
            !isFiniteScalar(pathV) ||
            !isFiniteScalar(pathRoll))
            return false;

        const T appliedU = std::max(minRadius, pathU);
        const auto offset = makePathOffsetFromState(pathS, appliedU, pathV);

        camera_vector_t<T, 2> orbitUv = camera_vector_t<T, 2>(T(0));
        T distance = T(0);
        if (!tryBuildOrbitFromPosition(targetPosition, targetPosition + offset, minDistance, maxDistance, orbitUv, distance))
            return false;
        if (!tryBuildSphericalPoseFromOrbit(targetPosition, orbitUv, distance, minDistance, maxDistance, outPosition, outOrientation, &distance))
            return false;

        if (!isNearlyZeroScalar(pathRoll, std::numeric_limits<T>::epsilon()))
        {
            const auto basis = getQuaternionBasisMatrix(outOrientation);
            const T rollCos = hlsl::cos(pathRoll);
            const T rollSin = hlsl::sin(pathRoll);
            const auto right = basis[0u] * rollCos + basis[1u] * rollSin;
            const auto up = basis[1u] * rollCos - basis[0u] * rollSin;
            outOrientation = makeQuaternionFromBasis(right, up, basis[2u]);
        }

        if (outAppliedDistance)
            *outAppliedDistance = distance;
        if (outOrbitUv)
            *outOrbitUv = orbitUv;
        return true;
    }

    template<typename T>
    static inline camera_vector_t<T, 3> rotateVectorByQuaternion(const camera_quaternion_t<T>& orientation, const camera_vector_t<T, 3>& vectorToRotate)
    {
        return normalizeQuaternion(orientation).transformVector(vectorToRotate, true);
    }

    template<typename T>
    static inline camera_vector_t<T, 3> projectWorldVectorToLocalBasis(
        const camera_vector_t<T, 3>& worldVector,
        const camera_vector_t<T, 3>& right,
        const camera_vector_t<T, 3>& up,
        const camera_vector_t<T, 3>& forward)
    {
        return camera_vector_t<T, 3>(
            dot(worldVector, right),
            dot(worldVector, up),
            dot(worldVector, forward));
    }

    template<typename T>
    static inline camera_vector_t<T, 3> transformLocalVectorToWorldBasis(
        const camera_vector_t<T, 3>& localVector,
        const camera_vector_t<T, 3>& right,
        const camera_vector_t<T, 3>& up,
        const camera_vector_t<T, 3>& forward)
    {
        return right * localVector.x + up * localVector.y + forward * localVector.z;
    }

    template<typename T>
    static inline camera_vector_t<T, 3> getQuaternionEulerRadians(const camera_quaternion_t<T>& orientation)
    {
        const auto q = normalizeQuaternion(orientation);
        const T x = q.data.x;
        const T y = q.data.y;
        const T z = q.data.z;
        const T w = q.data.w;

        const T pitch = hlsl::atan2(
            T(2) * (y * z + w * x),
            w * w - x * x - y * y + z * z);
        const T yaw = hlsl::asin(std::clamp(
            T(-2) * (x * z - w * y),
            T(-1),
            T(1)));
        const T roll = hlsl::atan2(
            T(2) * (x * y + w * z),
            w * w + x * x - y * y - z * z);

        return camera_vector_t<T, 3>(pitch, yaw, roll);
    }

    template<typename T>
    static inline camera_vector_t<T, 3> getQuaternionEulerDegrees(const camera_quaternion_t<T>& orientation)
    {
        const auto eulerRadians = getQuaternionEulerRadians(orientation);
        return camera_vector_t<T, 3>(
            degrees(eulerRadians.x),
            degrees(eulerRadians.y),
            degrees(eulerRadians.z));
    }

    template<typename T>
    static inline T getQuaternionAngularDistanceRadians(const camera_quaternion_t<T>& lhs, const camera_quaternion_t<T>& rhs)
    {
        const auto lhsNormalized = normalizeQuaternion(lhs);
        const auto rhsNormalized = normalizeQuaternion(rhs);
        const T orientationDot = std::clamp(
            static_cast<T>(hlsl::abs(dot(lhsNormalized.data, rhsNormalized.data))),
            T(0),
            T(1));
        return T(2) * hlsl::acos(orientationDot);
    }

    template<typename T>
    static inline T getQuaternionAngularDistanceDegrees(const camera_quaternion_t<T>& lhs, const camera_quaternion_t<T>& rhs)
    {
        return degrees(getQuaternionAngularDistanceRadians(lhs, rhs));
    }

    template<typename T>
    static inline bool tryComputePoseDelta(
        const camera_vector_t<T, 3>& lhsPosition,
        const camera_quaternion_t<T>& lhsOrientation,
        const camera_vector_t<T, 3>& rhsPosition,
        const camera_quaternion_t<T>& rhsOrientation,
        SCameraPoseDelta<T>& outDelta)
    {
        outDelta = {};

        const auto lhsNormalized = normalizeQuaternion(lhsOrientation);
        const auto rhsNormalized = normalizeQuaternion(rhsOrientation);
        if (!isFiniteVec3(lhsPosition) || !isFiniteVec3(rhsPosition) ||
            !isFiniteQuaternion(lhsNormalized) || !isFiniteQuaternion(rhsNormalized))
        {
            return false;
        }

        outDelta.position = length(lhsPosition - rhsPosition);
        outDelta.rotationDeg = getQuaternionAngularDistanceDegrees(lhsNormalized, rhsNormalized);
        return isFiniteScalar(outDelta.position) && isFiniteScalar(outDelta.rotationDeg);
    }

    template<typename T>
    static inline camera_quaternion_t<T> slerpQuaternion(const camera_quaternion_t<T>& lhs, const camera_quaternion_t<T>& rhs, const T alpha)
    {
        return camera_quaternion_t<T>::slerp(normalizeQuaternion(lhs), normalizeQuaternion(rhs), alpha);
    }

    template<typename T>
    static inline camera_quaternion_t<T> inverseQuaternion(const camera_quaternion_t<T>& q)
    {
        return inverse(q);
    }

    template<typename T>
    static inline camera_vector_t<T, 3> projectWorldVectorToLocalQuaternionFrame(
        const camera_quaternion_t<T>& orientation,
        const camera_vector_t<T, 3>& worldVector)
    {
        return rotateVectorByQuaternion(inverseQuaternion(orientation), worldVector);
    }

    template<typename T>
    static inline camera_matrix_t<T, 3, 3> getQuaternionBasisMatrix(const camera_quaternion_t<T>& orientation)
    {
        const auto q = normalizeQuaternion(orientation);
        return camera_matrix_t<T, 3, 3>(
            q.transformVector(camera_vector_t<T, 3>(T(1), T(0), T(0)), true),
            q.transformVector(camera_vector_t<T, 3>(T(0), T(1), T(0)), true),
            q.transformVector(camera_vector_t<T, 3>(T(0), T(0), T(1)), true));
    }

    template<typename T>
    static inline camera_vector_t<T, 3> getQuaternionEulerRadiansYXZ(const camera_quaternion_t<T>& orientation)
    {
        const auto basis = getQuaternionBasisMatrix(orientation);
        const T yaw = hlsl::atan2(basis[2][0], basis[2][2]);
        const T c2 = hlsl::length(camera_vector_t<T, 2>(basis[0][1], basis[1][1]));
        const T pitch = hlsl::atan2(-basis[2][1], c2);
        const T s1 = hlsl::sin(yaw);
        const T c1 = hlsl::cos(yaw);
        const T roll = hlsl::atan2(
            s1 * basis[1][2] - c1 * basis[1][0],
            c1 * basis[0][0] - s1 * basis[0][2]);
        return camera_vector_t<T, 3>(pitch, yaw, roll);
    }

    template<typename T>
    static inline camera_vector_t<T, 3> getQuaternionEulerDegreesYXZ(const camera_quaternion_t<T>& orientation)
    {
        const auto eulerRadians = getQuaternionEulerRadiansYXZ(orientation);
        return camera_vector_t<T, 3>(
            degrees(eulerRadians.x),
            degrees(eulerRadians.y),
            degrees(eulerRadians.z));
    }

    template<typename T>
    static inline camera_vector_t<T, 3> getCameraOrientationEulerRadians(const camera_quaternion_t<T>& orientation)
    {
        return getQuaternionEulerRadiansYXZ(orientation);
    }

    template<typename T>
    static inline camera_vector_t<T, 3> getCameraOrientationEulerDegrees(const camera_quaternion_t<T>& orientation)
    {
        return getQuaternionEulerDegreesYXZ(orientation);
    }

    template<typename T>
    static inline camera_vector_t<T, 3> getOrientationDeltaEulerRadiansYXZ(
        const camera_quaternion_t<T>& from,
        const camera_quaternion_t<T>& to)
    {
        const auto deltaQuat = inverseQuaternion(from) * normalizeQuaternion(to);
        return getQuaternionEulerRadiansYXZ(deltaQuat);
    }

    template<typename T>
    static inline camera_vector_t<T, 3> getWrappedEulerDistanceDegrees(
        const camera_vector_t<T, 3>& a,
        const camera_vector_t<T, 3>& b)
    {
        return camera_vector_t<T, 3>(
            getWrappedAngleDistanceDegrees(a.x, b.x),
            getWrappedAngleDistanceDegrees(a.y, b.y),
            getWrappedAngleDistanceDegrees(a.z, b.z));
    }

    template<typename T>
    static inline T getMaxVectorComponent(const camera_vector_t<T, 3>& value)
    {
        return std::max(value.x, std::max(value.y, value.z));
    }

    template<typename T>
    static inline camera_matrix_t<T, 4, 4> composeTransformMatrix(
        const camera_vector_t<T, 3>& translation,
        const camera_quaternion_t<T>& orientation,
        const camera_vector_t<T, 3>& scale = camera_vector_t<T, 3>(T(1)))
    {
        camera_matrix_t<T, 4, 4> output = camera_matrix_t<T, 4, 4>(1);
        const auto basis = getQuaternionBasisMatrix(orientation);
        output[0] = camera_vector_t<T, 4>(basis[0] * scale.x, T(0));
        output[1] = camera_vector_t<T, 4>(basis[1] * scale.y, T(0));
        output[2] = camera_vector_t<T, 4>(basis[2] * scale.z, T(0));
        output[3] = camera_vector_t<T, 4>(translation, T(1));
        return output;
    }

    template<typename T>
    static inline bool tryExtractRigidTransformComponents(
        const camera_matrix_t<T, 4, 4>& transform,
        SRigidTransformComponents<T>& outComponents)
    {
        outComponents.translation = camera_vector_t<T, 3>(transform[3].x, transform[3].y, transform[3].z);

        auto right = camera_vector_t<T, 3>(transform[0].x, transform[0].y, transform[0].z);
        auto up = camera_vector_t<T, 3>(transform[1].x, transform[1].y, transform[1].z);
        auto forward = camera_vector_t<T, 3>(transform[2].x, transform[2].y, transform[2].z);

        outComponents.scale = camera_vector_t<T, 3>(length(right), length(up), length(forward));

        if (!isFiniteVec3(outComponents.translation) || !isFiniteVec3(outComponents.scale))
            return false;

        constexpr T Epsilon = std::numeric_limits<T>::epsilon();
        if (outComponents.scale.x <= Epsilon || outComponents.scale.y <= Epsilon || outComponents.scale.z <= Epsilon)
            return false;

        right /= outComponents.scale.x;
        up /= outComponents.scale.y;
        forward /= outComponents.scale.z;
        if (!isOrthoBase(right, up, forward))
            return false;

        outComponents.orientation = makeQuaternionFromBasis(right, up, forward);
        return isFiniteQuaternion(outComponents.orientation);
    }

    template<typename T>
    static inline bool tryBuildRigidFrameFromTransform(
        const camera_matrix_t<T, 4, 4>& transform,
        camera_matrix_t<T, 4, 4>& outFrame,
        camera_quaternion_t<T>& outOrientation)
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
        const camera_matrix_t<T, 4, 4>& transform,
        camera_vector_t<T, 3>& outTranslation,
        camera_vector_t<T, 3>& outRotationEulerDegrees,
        camera_vector_t<T, 3>& outScale)
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

} // namespace nbl::hlsl

#endif // _C_CAMERA_MATH_UTILITIES_HPP_
