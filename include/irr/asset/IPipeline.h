#ifndef __IRR_I_PIPELINE_H_INCLUDED__
#define __IRR_I_PIPELINE_H_INCLUDED__

#include <utility>


#include "irr/core/IReferenceCounted.h"

namespace irr
{
namespace asset
{

template<typename LayoutType>
class IPipeline : public virtual core::IReferenceCounted
{
	public:
		enum E_PIPELINE_CREATION : uint32_t
		{
			EPC_DISABLE_OPTIMIZATIONS = 1<<0,
			EPC_ALLOW_DERIVATIVES = 1<<1,
			EPC_DERIVATIVE = 1<<2,
			EPC_VIEW_INDEX_FROM_DEVICE_INDEX = 1<<3,
			EPC_DISPATCH_BASE = 1<<4,
			EPC_DEFER_COMPILE_NV = 1<<5
		};

		inline const LayoutType* getLayout() const { return m_layout.get(); }

	protected:
		IPipeline(core::smart_refctd_ptr<LayoutType>&& _layout) :
			m_layout(std::move(_layout))
		{
		}
		virtual ~IPipeline() = default;

		core::smart_refctd_ptr<LayoutType> m_layout;
		bool m_disableOptimizations = false;
};

}
}

#endif