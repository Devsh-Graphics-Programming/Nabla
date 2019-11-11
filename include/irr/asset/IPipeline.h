#ifndef __IRR_I_PIPELINE_H_INCLUDED__
#define __IRR_I_PIPELINE_H_INCLUDED__

#include <utility>


#include "irr/core/IReferenceCounted.h"

namespace irr
{
namespace asset
{

namespace impl
{
	class IPipelineBase : public virtual core::IReferenceCounted
	{
		protected:
			IPipelineBase() = default;
			virtual ~IPipelineBase() = 0;
	};
}

template<typename LayoutType>
class IPipeline : public impl::IPipelineBase
{
	public:
		inline const LayoutType* getLayout() const { return m_layout.get(); }

	protected:
		IPipeline(core::smart_refctd_ptr<IPipeline<LayoutType> >&& _parent, core::smart_refctd_ptr<LayoutType>&& _layout) :
			m_parent(std::move(_parent)), m_layout(std::move(_layout))
		{
		}
		virtual ~IPipeline() = default;

		core::smart_refctd_ptr<IPipelineBase> m_parent;
		core::smart_refctd_ptr<LayoutType> m_layout;
		bool m_disableOptimizations = false;
		bool m_allowDerivatives = false;
};

}
}

#endif