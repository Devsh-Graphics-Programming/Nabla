#ifndef _NBL_I_QUAD_PROJECTION_HPP_
#define _NBL_I_QUAD_PROJECTION_HPP_

#include "ILinearProjection.hpp"

namespace nbl::core
{

/// @brief Interface for quad projections.
///
/// This projection transforms a vector into the model space of a perspective
/// quad defined by the pre-transform matrix and then projects it onto the quad
/// using the linear viewport transform.
///
/// A perspective quad projection is represented by:
/// - a pre-transform matrix
/// - a linear viewport transform matrix
///
/// The final projection matrix is the concatenation of those two transforms.
///
/// @note One perspective quad projection can represent a face quad of a CAVE-like system.
class IPerspectiveProjection : public ILinearProjection
{
public:
    /// @brief One quad projection entry described by a pretransform and a viewport projection.
    struct CProjection : ILinearProjection::CProjection
    {
        using base_t = ILinearProjection::CProjection;

        CProjection() = default;
        CProjection(const ILinearProjection::model_matrix_t& pretransform, ILinearProjection::concatenated_matrix_t viewport) 
        {
            setQuadTransform(pretransform, viewport); 
        }

        /// @brief Rebuild the concatenated quad projection from its authored components.
        inline void setQuadTransform(const ILinearProjection::model_matrix_t& pretransform, ILinearProjection::concatenated_matrix_t viewport)
        {
            auto concatenated = hlsl::mul(hlsl::CCameraMathUtilities::promoteAffine3x4To4x4(pretransform), viewport);
            base_t::setProjectionMatrix(concatenated);

            m_pretransform = pretransform;
            m_viewport = viewport;
        }

        /// @brief Return the authored pretransform applied before the viewport projection.
        inline const ILinearProjection::model_matrix_t& getPretransform() const { return m_pretransform; }
        /// @brief Return the authored viewport projection matrix stored for this quad.
        inline const ILinearProjection::concatenated_matrix_t& getViewportProjection() const { return m_viewport; }

    private:
        ILinearProjection::model_matrix_t m_pretransform = ILinearProjection::model_matrix_t(1);
        ILinearProjection::concatenated_matrix_t m_viewport = ILinearProjection::concatenated_matrix_t(1);
    };

protected:
    IPerspectiveProjection(core::smart_refctd_ptr<ICamera>&& camera)
        : ILinearProjection(core::smart_refctd_ptr(camera)) {}
    virtual ~IPerspectiveProjection() = default;
};

} // nbl::hlsl namespace

#endif // _NBL_I_QUAD_PROJECTION_HPP_
