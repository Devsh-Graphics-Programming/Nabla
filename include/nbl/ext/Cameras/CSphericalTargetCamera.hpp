#ifndef _C_SPHERICAL_TARGET_CAMERA_HPP_
#define _C_SPHERICAL_TARGET_CAMERA_HPP_

#include <algorithm>
#include "CCameraTargetRelativeUtilities.hpp"

namespace nbl::core
{

/// @brief Common base for target-relative cameras represented by target position, distance, and `orbitUv`.
///
/// Derived cameras keep the same target-relative storage but apply different
/// constraints and event policies in `manipulate(...)`.
class CSphericalTargetCamera : public ICamera
{
public:
    using base_t = ICamera;

    CSphericalTargetCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : base_t(), m_targetPosition(target), m_distance(SCameraTargetRelativeRigDefaults::InitialDistance),
          m_gimbal(typename base_t::CGimbal::base_t::SCreationParameters{
              .position = position,
              .orientation = hlsl::CCameraMathUtilities::makeIdentityQuaternion<hlsl::float64_t>()
          })
    {
        initFromPosition(position);
    }
    ~CSphericalTargetCamera() = default;

    inline bool setDistance(float d)
    {
        const auto clamped = std::clamp<float>(d, MinDistance, MaxDistance);
        const bool ok = clamped == d;
        if (m_distance == clamped)
            return ok;
        m_distance = clamped;
        applyPose();
        return ok;
    }

    inline void target(const hlsl::float64_t3& p)
    {
        if (m_targetPosition == p)
            return;
        m_targetPosition = p;
        applyPose();
    }
    inline hlsl::float64_t3 getTarget() const { return m_targetPosition; }

    inline float getDistance() const { return m_distance; }
    inline const hlsl::float64_t2& getOrbitUv() const { return m_orbitUv; }

    static inline constexpr float MinDistance = SCameraTargetRelativeTraits::MinDistance;
    static inline constexpr float MaxDistance = SCameraTargetRelativeTraits::DefaultMaxDistance;

    virtual uint32_t getCapabilities() const override
    {
        return base_t::SphericalTarget;
    }

    virtual bool tryGetSphericalTargetState(typename base_t::SphericalTargetState& out) const override
    {
        out.target = m_targetPosition;
        out.distance = m_distance;
        out.orbitUv = m_orbitUv;
        out.minDistance = MinDistance;
        out.maxDistance = MaxDistance;
        return true;
    }

    virtual bool trySetSphericalTarget(const hlsl::float64_t3& targetPosition) override
    {
        target(targetPosition);
        return true;
    }

    virtual bool trySetSphericalDistance(float distance) override
    {
        return setDistance(distance);
    }

protected:
    using SphericalBasis = SCameraTargetRelativeBasis;

    /// @brief Return the current canonical target-relative state stored by the spherical rig.
    inline SCameraTargetRelativeState currentTargetRelativeState() const
    {
        return {
            .target = m_targetPosition,
            .orbitUv = m_orbitUv,
            .distance = m_distance
        };
    }

    /// @brief Replace the stored target-relative state without touching the gimbal pose yet.
    inline void adoptTargetRelativeState(const SCameraTargetRelativeState& state)
    {
        m_targetPosition = state.target;
        m_orbitUv = state.orbitUv;
        m_distance = state.distance;
    }

    /// @brief Extract one rigid reference transform from the optional external override or the current gimbal pose.
    inline bool tryExtractReferenceTransform(CReferenceTransform& outReference, const hlsl::float64_t4x4* referenceFrame)
    {
        return m_gimbal.extractReferenceTransform(&outReference, referenceFrame);
    }

    /// @brief Resolve the current target-relative state from one rigid reference position around the current target.
    inline bool tryResolveReferenceTargetRelativeState(const CReferenceTransform& reference, SCameraTargetRelativeState& outState) const
    {
        return CCameraTargetRelativeUtilities::tryBuildTargetRelativeStateFromPosition(
            m_targetPosition,
            hlsl::float64_t3(reference.frame[3]),
            MinDistance,
            MaxDistance,
            outState);
    }

    /// @brief Resolve the top-down yaw encoded by a rigid reference orientation.
    static inline double resolveTopDownYawFromReference(const CReferenceTransform& reference, const double fallbackYaw)
    {
        const auto basis = hlsl::CCameraMathUtilities::getQuaternionBasisMatrix(reference.orientation);
        const auto planarUp = hlsl::float64_t2(basis[1].x, basis[1].y);
        constexpr auto Epsilon = static_cast<hlsl::float64_t>(SCameraToolingThresholds::TinyScalarEpsilon);
        if (!hlsl::CCameraMathUtilities::isNearlyZeroVector(planarUp, Epsilon))
            return hlsl::atan2(planarUp.y, planarUp.x);

        const auto planarRight = hlsl::float64_t2(basis[0].x, basis[0].y);
        if (!hlsl::CCameraMathUtilities::isNearlyZeroVector(planarRight, Epsilon))
            return hlsl::atan2(planarRight.x, -planarRight.y);

        return fallbackYaw;
    }

    /// @brief Project one rigid reference pose onto the legal top-down state manifold around the current target.
    inline bool tryResolveReferenceTopDownState(const CReferenceTransform& reference, SCameraTargetRelativeState& outState) const
    {
        const auto offset = hlsl::float64_t3(reference.frame[3]) - m_targetPosition;
        const auto distance = hlsl::length(offset);
        if (!hlsl::CCameraMathUtilities::isFiniteScalar(distance) ||
            distance <= static_cast<hlsl::float64_t>(SCameraToolingThresholds::TinyScalarEpsilon))
        {
            return false;
        }

        outState = currentTargetRelativeState();
        outState.distance = static_cast<float>(std::clamp(
            distance,
            static_cast<hlsl::float64_t>(MinDistance),
            static_cast<hlsl::float64_t>(MaxDistance)));
        outState.orbitUv.x = resolveTopDownYawFromReference(reference, m_orbitUv.x);
        outState.orbitUv.y = SCameraTargetRelativeRigDefaults::TopDownPitchRad;
        return true;
    }

    /// @brief Project one rigid reference pose onto the legal fixed-angle isometric manifold around the current target.
    inline bool tryResolveReferenceIsometricState(const CReferenceTransform& reference, SCameraTargetRelativeState& outState) const
    {
        if (!tryResolveReferenceTargetRelativeState(reference, outState))
            return false;

        outState.orbitUv = hlsl::float64_t2(
            SCameraTargetRelativeRigDefaults::IsometricYawRad,
            SCameraTargetRelativeRigDefaults::IsometricPitchRad);
        return true;
    }

    inline SphericalBasis computeBasis(const hlsl::float64_t2& orbitUv, float distance) const
    {
        SphericalBasis basis;
        const SCameraTargetRelativeState state = {
            .target = m_targetPosition,
            .orbitUv = orbitUv,
            .distance = distance
        };
        if (!CCameraTargetRelativeUtilities::tryBuildTargetRelativeBasis(state, MinDistance, MaxDistance, basis))
            return basis;
        return basis;
    }

    inline void initFromPosition(const hlsl::float64_t3& position)
    {
        SCameraTargetRelativeState state = {};
        if (!CCameraTargetRelativeUtilities::tryBuildTargetRelativeStateFromPosition(m_targetPosition, position, MinDistance, MaxDistance, state))
        {
            m_distance = MinDistance;
            m_orbitUv = hlsl::float64_t2(0.0);
            return;
        }

        m_distance = state.distance;
        m_orbitUv = state.orbitUv;
    }

    inline void applyPlanarTargetTranslation(const hlsl::float64_t3& deltaTranslation, const SphericalBasis& basis)
    {
        if (!hlsl::CCameraMathUtilities::hasPlanarDeltaXY(deltaTranslation, static_cast<hlsl::float64_t>(SCameraToolingThresholds::TinyScalarEpsilon)))
            return;

        m_targetPosition += hlsl::CCameraMathUtilities::transformLocalVectorToWorldBasis(
            hlsl::float64_t3(deltaTranslation.x, deltaTranslation.y, 0.0),
            basis.right,
            basis.up,
            basis.forward);
    }

    inline bool applyPose()
    {
        const SCameraTargetRelativeState state = {
            .target = m_targetPosition,
            .orbitUv = m_orbitUv,
            .distance = m_distance
        };
        SCameraTargetRelativePose pose = {};
        if (!CCameraTargetRelativeUtilities::tryBuildTargetRelativePoseFromState(state, MinDistance, MaxDistance, pose))
            return false;
        m_distance = static_cast<float>(pose.appliedDistance);

        m_gimbal.begin();
        {
            m_gimbal.setPosition(pose.position);
            m_gimbal.setOrientation(pose.orientation);
        }
        m_gimbal.end();

        const bool manipulated = bool(m_gimbal.getManipulationCounter());
        if (manipulated)
            m_gimbal.updateView();

        return manipulated;
    }

    hlsl::float64_t3 m_targetPosition;
    float m_distance;
    typename base_t::CGimbal m_gimbal;
    hlsl::float64_t2 m_orbitUv = hlsl::float64_t2(0.0);
};

}

#endif

