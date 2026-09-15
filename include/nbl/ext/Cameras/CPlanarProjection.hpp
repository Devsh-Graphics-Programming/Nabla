#ifndef _NBL_C_PLANAR_PROJECTION_HPP_
#define _NBL_C_PLANAR_PROJECTION_HPP_

#include "IPlanarProjection.hpp"
#include "IRange.hpp"

namespace nbl::core
{
	/// @brief Range-backed concrete implementation of `IPlanarProjection`.
	///
	/// The template owns a caller-selected contiguous container of planar
	/// projection entries together with their viewport-local binding layouts.
	template<ContiguousGeneralPurposeRangeOf<IPlanarProjection::CProjection> ProjectionsRange>
	class CPlanarProjection : public IPlanarProjection
	{
	public:
		virtual ~CPlanarProjection() = default;

		/// @brief Create a planar projection wrapper only when a valid camera instance is available.
		inline static core::smart_refctd_ptr<CPlanarProjection> create(core::smart_refctd_ptr<ICamera>&& camera)
		{
			if (!camera)
				return nullptr;

			return core::smart_refctd_ptr<CPlanarProjection>(new CPlanarProjection(core::smart_refctd_ptr(camera)), core::dont_grab);
		}

		/// @brief Return the number of stored planar projection entries.
		virtual uint32_t getLinearProjectionCount() const override
		{
			return static_cast<uint32_t>(m_projections.size());
		}

		/// @brief Return one stored planar projection entry through the linear base interface.
		virtual const ILinearProjection::CProjection& getLinearProjection(uint32_t index) const override
		{
			assert(index < m_projections.size());
			return m_projections[index];
		}

		/// @brief Expose mutable access to the owned planar projection range.
		inline ProjectionsRange& getPlanarProjections()
		{
			return m_projections;
		}

	protected:
		CPlanarProjection(core::smart_refctd_ptr<ICamera>&& camera)
			: IPlanarProjection(core::smart_refctd_ptr(camera)) {}

		ProjectionsRange m_projections;
	};

} // nbl::hlsl namespace

#endif // _NBL_C_PLANAR_PROJECTION_HPP_
