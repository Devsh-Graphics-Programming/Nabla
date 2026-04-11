#ifndef _C_ORBIT_CAMERA_HPP_
#define _C_ORBIT_CAMERA_HPP_

#include <algorithm>
#include <cmath>
#include "CSphericalTargetCamera.hpp"

namespace nbl::core
{

/// @brief Target-relative camera with state `(target, orbitUv, distance)`.
///
/// Runtime input updates only orbit yaw, orbit pitch, and camera distance.
/// The target position remains unchanged during `manipulate(...)`.
class COrbitCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    COrbitCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : base_t(position, target)
    {
        m_distance = std::clamp<float>(hlsl::length(m_targetPosition - position), MinDistance, MaxDistance);
        applyPose();
    }
    ~COrbitCamera() = default;

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

    /// @brief Apply one frame of orbit-angle and distance input around the current target.
    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame = nullptr) override
    {
        if (virtualEvents.empty() && !referenceFrame)
            return false;

        if (referenceFrame)
        {
            CReferenceTransform reference = {};
            SCameraTargetRelativeState resolvedState = {};
            if (!tryExtractReferenceTransform(reference, referenceFrame) ||
                !tryResolveReferenceTargetRelativeState(reference, resolvedState))
            {
                return false;
            }

            adoptTargetRelativeState(resolvedState);
        }

        const auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);
        const auto deltaTranslation = scaleVirtualTranslation(impulse.dVirtualTranslate);
        const double deltaDistance = scaleUnscaledVirtualTranslation(impulse.dVirtualTranslate.z);

        m_orbitUv += hlsl::float64_t2(deltaTranslation.y, deltaTranslation.x);
   
        m_distance = std::clamp<float>(m_distance + static_cast<float>(deltaDistance), MinDistance, MaxDistance);

        return applyPose();
    }

    virtual uint32_t getAllowedVirtualEvents() const override
    {
        return AllowedVirtualEvents;
    }

    virtual CameraKind getKind() const override
    {
        return CameraKind::Orbit;
    }

    virtual std::string_view getIdentifier() const override
    {
        return "Orbit Camera";
    }

    static inline constexpr float MinDistance = base_t::MinDistance;
    static inline constexpr float MaxDistance = base_t::MaxDistance;

    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate;
};

}

#endif // _C_ORBIT_CAMERA_HPP_
