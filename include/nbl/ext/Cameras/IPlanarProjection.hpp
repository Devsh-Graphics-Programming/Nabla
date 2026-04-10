#ifndef _NBL_I_PLANAR_PROJECTION_HPP_
#define _NBL_I_PLANAR_PROJECTION_HPP_

#include "nbl/builtin/hlsl/math/thin_lens_projection.hlsl"

#include "IGimbalBindingLayout.hpp"
#include "ILinearProjection.hpp"

namespace nbl::core
{

/// @brief Linear projection wrapper for one camera-facing planar viewport.
///
/// The projection stores viewport-local binding layouts. Runtime input
/// processing is handled by `CGimbalInputBinder`.
class IPlanarProjection : public ILinearProjection
{
public:
    /// @brief One perspective or orthographic projection entry plus its viewport-local bindings.
    struct CProjection : public ILinearProjection::CProjection
    {
        using base_t = ILinearProjection::CProjection;

        /// @brief Stable runtime classification of supported planar projection parameterizations.
        enum ProjectionType : uint8_t
        {
            Perspective,
            Orthographic,

            Count
        };

        template<ProjectionType T, typename... Args>
        static CProjection create(Args&&... args)
        requires (T != Count)
        {
            CProjection output;

            if constexpr (T == Perspective) output.setPerspective(std::forward<Args>(args)...);
            else if (T == Orthographic) output.setOrthographic(std::forward<Args>(args)...);

            return output;
        }

        CProjection(const CProjection& other) = default;
        CProjection(CProjection&& other) noexcept = default;

        /// @brief Authored parameter bundle stored by one planar projection entry.
        struct ProjectionParameters
        {
            ProjectionType m_type;

            union PlanarParameters
            {
                struct
                {
                    float fov;
                } perspective;

                struct
                {
                    float orthoWidth;
                } orthographic;

                PlanarParameters() {}
                ~PlanarParameters() {}
            } m_planar;

            float m_zNear;
            float m_zFar;
        };

        /// @brief Rebuild the concrete projection matrix from the stored parameters.
        inline void update(bool leftHanded, float aspectRatio)
        {
            switch (m_parameters.m_type)
            {
                case Perspective:
                {
                    const auto& fov = m_parameters.m_planar.perspective.fov;

                    if (leftHanded)
                        base_t::setProjectionMatrix(hlsl::buildProjectionMatrixPerspectiveFovLH<hlsl::float64_t>(hlsl::radians(fov), aspectRatio, m_parameters.m_zNear, m_parameters.m_zFar));
                    else
                        base_t::setProjectionMatrix(hlsl::buildProjectionMatrixPerspectiveFovRH<hlsl::float64_t>(hlsl::radians(fov), aspectRatio, m_parameters.m_zNear, m_parameters.m_zFar));
                } break;

                case Orthographic:
                {
                    const auto& orthoW = m_parameters.m_planar.orthographic.orthoWidth;
                    const auto viewHeight = orthoW / aspectRatio;

                    if (leftHanded)
                        base_t::setProjectionMatrix(hlsl::buildProjectionMatrixOrthoLH<hlsl::float64_t>(orthoW, viewHeight, m_parameters.m_zNear, m_parameters.m_zFar));
                    else
                        base_t::setProjectionMatrix(hlsl::buildProjectionMatrixOrthoRH<hlsl::float64_t>(orthoW, viewHeight, m_parameters.m_zNear, m_parameters.m_zFar));
                } break;
            }
        }

        /// @brief Switch the entry to perspective mode and store its authored parameters.
        inline void setPerspective(float zNear = 0.1f, float zFar = 100.f, float fov = 60.f)
        {
            m_parameters.m_type = Perspective;
            m_parameters.m_planar.perspective.fov = fov;
            m_parameters.m_zNear = zNear;
            m_parameters.m_zFar = zFar;
        }

        /// @brief Switch the entry to orthographic mode and store its authored parameters.
        inline void setOrthographic(float zNear = 0.1f, float zFar = 100.f, float orthoWidth = 10.f)
        {
            m_parameters.m_type = Orthographic;
            m_parameters.m_planar.orthographic.orthoWidth = orthoWidth;
            m_parameters.m_zNear = zNear;
            m_parameters.m_zFar = zFar;
        }

        /// @brief Return the authored planar projection parameters.
        inline const ProjectionParameters& getParameters() const { return m_parameters; }
        /// @brief Return the viewport-local input binding layout stored next to this projection entry.
        inline const ui::IGimbalBindingLayout& getInputBinding() const { return m_inputBinding; }
        /// @brief Return mutable access to the viewport-local input binding layout.
        inline ui::IGimbalBindingLayout& getInputBinding() { return m_inputBinding; }
    private:
        CProjection() = default;
        ProjectionParameters m_parameters;
        ui::CGimbalBindingLayoutStorage m_inputBinding;
    };

protected:
    IPlanarProjection(core::smart_refctd_ptr<ICamera>&& camera)
        : ILinearProjection(std::move(camera)) {}
    virtual ~IPlanarProjection() = default;
};

} // namespace nbl::core

#endif // _NBL_I_PLANAR_PROJECTION_HPP_
