#ifndef __IRR_I_GPU_DESCRIPTOR_SET_H_INCLUDED__
#define __IRR_I_GPU_DESCRIPTOR_SET_H_INCLUDED__

#include "irr/asset/IDescriptorSet.h"

#include "IGPUBuffer.h"
#include "irr/video/IGPUBufferView.h"
#include "irr/video/IGPUImageView.h"
#include "irr/video/IGPUSampler.h"
#include "irr/video/IGPUDescriptorSetLayout.h"

namespace irr
{
namespace video
{

//! GPU Version of Descriptor Set
/*
	@see IDescriptorSet
*/

class IGPUDescriptorSet : public asset::IDescriptorSet<const IGPUDescriptorSetLayout>
{
	public:
		using asset::IDescriptorSet<const IGPUDescriptorSetLayout>::IDescriptorSet;

	protected:
		virtual ~IGPUDescriptorSet() = default;
};

}
}

#endif