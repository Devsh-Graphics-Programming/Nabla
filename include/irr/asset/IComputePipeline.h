#ifndef __IRR_I_COMPUTE_PIPELINE_H_INCLUDED__
#define __IRR_I_COMPUTE_PIPELINE_H_INCLUDED__

#include "irr/asset/IPipeline.h"
#include <utility>

namespace irr
{
namespace asset
{

template<typename SpecShaderType, typename LayoutType>
class IComputePipeline : public IPipeline<IComputePipeline,LayoutType>
{
	protected:
		IComputePipeline(
			core::smart_refctd_ptr<IComputePipeline>&& _parent,
			core::smart_refctd_ptr<LayoutType>&& _layout,
			core::smart_refctd_ptr<SpecShaderType>&& _cs
		) : IPipeline<IComputePipeline,LayoutType>(std::move(_parent),std::move(_layout)),
			m_shader(std::move(_cs))
		{}
		virtual ~IComputePipeline() = default;

		core::smart_refctd_ptr<SpecShaderType> m_shader;
};

}
}


#endif