#ifndef __IRR_I_PIPELINE_H_INCLUDED__
#define __IRR_I_PIPELINE_H_INCLUDED__

#include <utility>

#include "irr/core/IReferenceCounted.h"

namespace irr
{
namespace asset
{

struct DrawArraysIndirectCommand_t
{
	uint32_t  count;
	uint32_t  instanceCount;
	uint32_t  first;
	uint32_t  baseInstance;
};

struct DrawElementsIndirectCommand_t
{
	uint32_t count;
	uint32_t instanceCount;
	uint32_t firstIndex;
	uint32_t baseVertex;
	uint32_t baseInstance;
};

struct DispatchIndirectCommand_t
{
	uint32_t  num_groups_x;
	uint32_t  num_groups_y;
	uint32_t  num_groups_z;
};

//! Interface class for for concreting graphics and compute pipelines
/*
	A pipeline refers to a succession of fixed stages 
	through which a data input flows; each stage processes 
	the incoming data and hands it over to the next stage. 
	The final product will be either a 2D raster drawing image 
	(the graphics pipeline) or updated resources (buffers or images) 
	with computational logic and calculations (the compute pipeline).

	Vulkan supports two types of pipeline:
	
	- graphics pipeline
	- compute pipeline
*/

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