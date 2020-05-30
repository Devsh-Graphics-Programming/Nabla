#include "irr/asset/CCPUMeshPacker.h"
#include "irrlicht.h"

namespace irr 
{ 
namespace asset 
{

std::optional<std::pair<ICPUMeshBuffer*, DrawElementsIndirectCommand_t>> CCPUMeshPacker::packMeshes(const core::vector<ICPUMeshBuffer*>& meshBuffers)
{
	if (meshBuffers.empty())
		return {};

	/*
	Requirements for input mesh buffers:
		- attributes bound to the same binding must have identical format
		- all meshbufers have indexed triangle list (temporary)
	*/

	//validation
	for (auto meshBuffer : meshBuffers)
	{
		auto* pipeline = meshBuffer->getPipeline();

		if (meshBuffer->getIndexBufferBinding()->buffer.get() == nullptr || 
			pipeline->getPrimitiveAssemblyParams().primitiveType == EPT_TRIANGLE_LIST)
		{
			_IRR_DEBUG_BREAK_IF(true);
			return {};
		}
	}

	SVertexInputParams outVtxInputParams;

	//set attributes and bindings of output mesh buffer
	for (const auto& meshBuffer : meshBuffers)
	{
		const auto& currMBVtxInputParams = meshBuffer->getPipeline()->getVertexInputParams();

		if (currMBVtxInputParams.enabledAttribFlags != outVtxInputParams.enabledAttribFlags)
		{
			const uint16_t oldFlags = outVtxInputParams.enabledAttribFlags;
			outVtxInputParams.enabledAttribFlags |= currMBVtxInputParams.enabledAttribFlags;
			const uint16_t flagDiff = outVtxInputParams.enabledAttribFlags ^ oldFlags;

			//if output mesh buffer doesn't have given attrib enabled but currently processed mesh buffer has, enable this attribute for output mesh buffer and set its format and binding
			for (uint16_t attrBit = 0x0001, location = 0; location < 16; attrBit <<= 1, location++)
			{
				if (flagDiff & attrBit)
				{
					outVtxInputParams.attributes[location].format = currMBVtxInputParams.attributes[location].format;
					outVtxInputParams.attributes[location].binding = location;
					outVtxInputParams.attributes[location].relativeOffset = 0;

					outVtxInputParams.bindings[location].stride = getTexelOrBlockBytesize(static_cast<E_FORMAT>(outVtxInputParams.attributes[location].format));
					outVtxInputParams.bindings[location].inputRate = currMBVtxInputParams.bindings[currMBVtxInputParams.attributes[location].binding].inputRate;
				}
			}
		}
	}

	//validation
	//check if formats of attributes with the same bindings are identical
	for (size_t i = 1; i < meshBuffers.size(); i++)
	{
		const auto& currMBVtxInputParams = meshBuffers[i]->getPipeline()->getVertexInputParams();

		for (uint16_t attrBit = 0x0001, location = 0; location < 16; attrBit <<= 1, location++)
		{
			if (!(attrBit & currMBVtxInputParams.enabledAttribFlags))
				continue;

			assert(!(attrBit & outVtxInputParams.enabledAttribFlags)); //imposibru

			if (currMBVtxInputParams.attributes[location].format != outVtxInputParams.attributes[location].format ||
				currMBVtxInputParams.bindings[currMBVtxInputParams.attributes[location].binding].inputRate != outVtxInputParams.bindings[location].inputRate)
			{
				_IRR_DEBUG_BREAK_IF(true);
				return {};
			}
		}
	}

	//outdated
	/*const size_t vtxCnt = std::accumulate(
		meshBuffers.begin(), meshBuffers.end(), 0ull,
		[] (size_t sum, core::vector<ICPUMeshBuffer*>::iterator mBuff) -> size_t
		{ 
			return sum + (*mBuff)->calcVertexCount();
		});
	*/

	const size_t idxCnt = std::accumulate(
		meshBuffers.begin(), meshBuffers.end(), 0ull,
		[](size_t sum, core::vector<ICPUMeshBuffer*>::iterator mBuff) -> size_t
		{
			return sum + (*mBuff)->getIndexCount();
		});

	if (idxCnt > std::numeric_limits<uint16_t>::max())
	{
		//TODO: create multiple draw calls in this case
		_IRR_DEBUG_BREAK_IF(true);
		return {};
	}

	//TODO: alignment
	//SBufferBinding<ICPUBuffer> packedVtxBuff{ 0u, core::make_smart_refctd_ptr<ICPUBuffer*>(vtxSize * vtxCnt) };
	SBufferBinding<ICPUBuffer> packedIdxBuff{ 0u, core::make_smart_refctd_ptr<ICPUBuffer*>(sizeof(uint16_t) * idxCnt) };
	constexpr E_INDEX_TYPE packedMeshIndexType = E_INDEX_TYPE::EIT_16BIT;


	return {};
}

}
}