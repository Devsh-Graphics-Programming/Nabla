#ifndef _NBL_C_LINEAR_PROJECTION_HPP_
#define _NBL_C_LINEAR_PROJECTION_HPP_

#include "ILinearProjection.hpp"
#include "IRange.hpp"

namespace nbl::core
{
	/// @brief Range-backed concrete implementation of `ILinearProjection`.
	///
	/// The template owns a caller-selected contiguous container of linear projection
	/// entries and exposes it through the generic `ILinearProjection` interface.
	template<ContiguousGeneralPurposeRangeOf<ILinearProjection::CProjection> ProjectionsRange>
	class CLinearProjection : public ILinearProjection
	{
	public:
		using ILinearProjection::ILinearProjection;

		CLinearProjection() = default;

		/// @brief Create a projection wrapper only when a valid camera instance is available.
		inline static core::smart_refctd_ptr<CLinearProjection> create(core::smart_refctd_ptr<ICamera>&& camera)
		{
			if (!camera)
				return nullptr;

			return core::smart_refctd_ptr<CLinearProjection>(new CLinearProjection(core::smart_refctd_ptr(camera)), core::dont_grab);
		}

		/// @brief Return the number of stored linear projection entries.
		virtual uint32_t getLinearProjectionCount() const override
		{
			return static_cast<uint32_t>(m_projections.size());
		}

		/// @brief Return one stored projection entry by index.
		virtual const CProjection& getLinearProjection(uint32_t index) const override
		{
			assert(index < m_projections.size());
			return m_projections[index];
		}

		/// @brief Expose mutable access to the owned projection range.
		inline std::span<CProjection> getLinearProjections()
		{
			return std::span<CProjection>(m_projections.data(), m_projections.size());
		}

	private:
		CLinearProjection(core::smart_refctd_ptr<ICamera>&& camera)
			: ILinearProjection(core::smart_refctd_ptr(camera)) {}
		virtual ~CLinearProjection() = default;

		ProjectionsRange m_projections;
	};

} // nbl::hlsl namespace

#endif // _NBL_C_LINEAR_PROJECTION_HPP_
