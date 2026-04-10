#ifndef _NBL_I_LINEAR_PROJECTION_HPP_
#define _NBL_I_LINEAR_PROJECTION_HPP_

#include <optional>

#include "nbl/core/decl/smart_refctd_ptr.h"
#include "IProjection.hpp"
#include "ICamera.hpp"

namespace nbl::core
{

/// @brief Interface for any custom linear projection transformation.
///
/// Matrix elements are already evaluated scalars referencing a camera.
/// This covers perspective, orthographic, oblique, axonometric, and shear projections.
class ILinearProjection : virtual public core::IReferenceCounted
{
protected:
    ILinearProjection(core::smart_refctd_ptr<ICamera>&& camera)
        : m_camera(std::move(camera)) {}
    virtual ~ILinearProjection() = default;

    core::smart_refctd_ptr<ICamera> m_camera;
public:
    /// @brief World transform type expected by the linear projection helpers.
    using model_matrix_t = typename ICamera::CGimbal::model_matrix_t;

    /// @brief Matrix type used for fully concatenated linear transforms.
    using concatenated_matrix_t = hlsl::float64_t4x4;

    /// @brief Optional inverse of a concatenated transform when the matrix is not singular.
    using inv_concatenated_matrix_t = std::optional<hlsl::float64_t4x4>;

    /// @brief One concrete linear projection matrix together with cached inverse metadata.
    struct CProjection : public IProjection
    {
        using IProjection::IProjection;
        using projection_matrix_t = concatenated_matrix_t;
        using inv_projection_matrix_t = inv_concatenated_matrix_t;

        CProjection() : CProjection(projection_matrix_t(1)) {}
        CProjection(const projection_matrix_t& matrix) { setProjectionMatrix(matrix); }

        /// @brief Returns P (Projection matrix)
        inline const projection_matrix_t& getProjectionMatrix() const { return m_projectionMatrix; }

        /// @brief Returns P⁻¹ (Inverse of Projection matrix) *if it exists*
        inline const inv_projection_matrix_t& getInvProjectionMatrix() const { return m_invProjectionMatrix; }

        inline const std::optional<bool>& isProjectionLeftHanded() const { return m_isProjectionLeftHanded; }
        inline bool isProjectionSingular() const { return m_isProjectionSingular; }
        virtual ProjectionType getProjectionType() const override { return ProjectionType::Linear; }

        virtual void project(const projection_vector_t& vecToProjectionSpace, projection_vector_t& output) const override
        {
            output = hlsl::mul(m_projectionMatrix, vecToProjectionSpace);
        }

        virtual bool unproject(const projection_vector_t& vecFromProjectionSpace, projection_vector_t& output) const override
        {
            if (m_isProjectionSingular)
                return false;

            output = hlsl::mul(m_invProjectionMatrix.value(), vecFromProjectionSpace);

            return true;
        }

    protected:
        /// @brief Replace the projection matrix and rebuild cached handedness and inverse information.
        inline void setProjectionMatrix(const projection_matrix_t& matrix)
        {
            m_projectionMatrix = matrix;
            const auto det = hlsl::determinant(m_projectionMatrix);

            // we will allow you to lose a dimension since such a projection itself *may* 
            // be valid, however then you cannot un-project because the inverse doesn't exist!
            m_isProjectionSingular = not det;

            if (m_isProjectionSingular)
            {
                m_isProjectionLeftHanded = std::nullopt;
                m_invProjectionMatrix = std::nullopt;
            }
            else
            {
                m_isProjectionLeftHanded = det < 0.0;
                m_invProjectionMatrix = hlsl::inverse(m_projectionMatrix);
            }
        }

    private:
        projection_matrix_t m_projectionMatrix;
        inv_projection_matrix_t m_invProjectionMatrix;
        std::optional<bool> m_isProjectionLeftHanded;
        bool m_isProjectionSingular;
    };

    /// @brief Return the number of linear projection entries owned by the concrete wrapper.
    virtual uint32_t getLinearProjectionCount() const = 0;
    /// @brief Return one linear projection entry by index.
    virtual const CProjection& getLinearProjection(uint32_t index) const = 0;
    
    /// @brief Replace the camera referenced by this projection wrapper.
    inline bool setCamera(core::smart_refctd_ptr<ICamera>&& camera)
    {
        if (camera)
        {
            m_camera = camera;
            return true;
        }

        return false;
    }

    /// @brief Return the camera referenced by this projection wrapper.
    inline ICamera* getCamera()
    {
        return m_camera.get();
    }

    /// @brief Compute the model-view matrix.
    ///
    /// @param model World TRS matrix.
    /// @return The model-view matrix.
    inline concatenated_matrix_t getMV(const model_matrix_t& model) const
    {
        const auto& v = m_camera->getGimbal().getViewMatrix();
        return hlsl::mul(hlsl::getMatrix3x4As4x4(v), hlsl::getMatrix3x4As4x4(model));
    }

    /// @brief Compute the model-view-projection matrix from a model matrix.
    ///
    /// @param projection Linear projection.
    /// @param model World TRS matrix.
    /// @return The model-view-projection matrix.
    inline concatenated_matrix_t getMVP(const CProjection& projection, const model_matrix_t& model) const
    {
        const auto& v = m_camera->getGimbal().getViewMatrix();
        const auto& p = projection.getProjectionMatrix();
        auto mv = hlsl::mul(hlsl::getMatrix3x4As4x4(v), hlsl::getMatrix3x4As4x4(model));
        return hlsl::mul(p, mv);
    }

    /// @brief Compute the model-view-projection matrix from a model-view matrix.
    ///
    /// @param projection Linear projection.
    /// @param mv Model-view matrix.
    /// @return The model-view-projection matrix.
    inline concatenated_matrix_t getMVP(const CProjection& projection, const concatenated_matrix_t& mv) const
    {
        const auto& p = projection.getProjectionMatrix();
        return hlsl::mul(p, mv);
    }

    /// @brief Compute the inverse model-view matrix.
    ///
    /// @param model World TRS matrix.
    /// @return The inverse model-view matrix when it exists, otherwise `std::nullopt`.
    inline inv_concatenated_matrix_t getMVInverse(const model_matrix_t& model) const
    {
        const auto mv = getMV(model);
        if (auto det = hlsl::determinant(mv); det)
            return hlsl::inverse(mv);
        return std::nullopt;
    }

    /// @brief Compute the inverse model-view-projection matrix.
    ///
    /// @param projection Linear projection.
    /// @param model World TRS matrix.
    /// @return The inverse model-view-projection matrix when it exists, otherwise `std::nullopt`.
    inline inv_concatenated_matrix_t getMVPInverse(const CProjection& projection, const model_matrix_t& model) const
    {
        const auto mvp = getMVP(projection, model);
        if (auto det = hlsl::determinant(mvp); det)
            return hlsl::inverse(mvp);
        return std::nullopt;
    }
};

} // namespace nbl::core

#endif // _NBL_I_LINEAR_PROJECTION_HPP_
