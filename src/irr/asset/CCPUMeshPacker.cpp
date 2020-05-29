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

	//define SVertexInputParams of packed mesh buffer
	SVertexInputParams outVtxInputParams = meshBuffers[0]->getPipeline()->getVertexInputParams();
	outVtxInputParams.enabledBindingFlags = 0x0001; // < binding for unified VBO
	memset(outVtxInputParams.bindings, 0, sizeof(SVertexInputParams::bindings));

	auto findAttribIndexBoundToN = [](const uint32_t N, const SVertexInputParams& attrParams) -> int
	{
		if (N > 15)
			return -1;

		for (uint32_t i = 0; i < SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT; i++)
		{
			const SVertexInputAttribParams& vtxInputParam = attrParams.attributes[i];
			if (vtxInputParam.binding == N)
				return i;
		}

		return -1;
	};

	uint16_t sharedVtxAttribsMask = outVtxInputParams.enabledAttribFlags; //< attributes that are enabled for EVERY input mesh buffers ("alive" attributes)
	//establish which attrib bindings are shared among all mesh buffers ("alive" attributes)
	for (size_t i = 1; i < meshBuffers.size(); i++)
	{
		const auto& currMBVtxInputParams = meshBuffers[i]->getPipeline()->getVertexInputParams();

		if (currMBVtxInputParams.enabledAttribFlags != outVtxInputParams.enabledAttribFlags)
		{
			//if currently processed mesh buffer has given attrib enabled but packed mesh buffer don't, then eneble this attrib in packed mesh buffer
			outVtxInputParams.enabledAttribFlags |= currMBVtxInputParams.enabledAttribFlags;

			for (uint16_t attrBit = 0x0001, binding = 0; binding < 16; attrBit <<= 1, binding++)
			{
				//if currently processed mesh buffer doesn't have given attrib enabled but all of the previous mesh buffers had, then remove this attrib from `sharedVtxAttribsMask`
				if (!(attrBit & currMBVtxInputParams.enabledAttribFlags) && (attrBit & sharedVtxAttribsMask))
					sharedVtxAttribsMask ^= attrBit;

				//set format of new attribute
				const int attrBindingIndex = findAttribIndexBoundToN(binding, currMBVtxInputParams);
				assert(attrBindingIndex != -1); //imposibru 

				const asset::E_FORMAT formatOfNewAttr = static_cast<asset::E_FORMAT>(currMBVtxInputParams.attributes[attrBindingIndex].format);

				//assign to first matching attr param
				SVertexInputAttribParams* attrToAssign = outVtxInputParams.attributes;
				for (uint32_t i = 0; i < SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT; i++)
				{
					if (attrToAssign->format == asset::EF_UNKNOWN)
					{
						attrToAssign->format = formatOfNewAttr;
						break;
					}

					attrToAssign++;
				}
			}
		}
	}

	//validation
	//check if formats of attributes with the same bindings are identical
	for (size_t i = 1; i < meshBuffers.size(); i++)
	{
		const auto& vtxInputParams = meshBuffers[i]->getPipeline()->getVertexInputParams();

		for (uint16_t attrBit = 0x0001, binding = 0; binding < 16; attrBit <<= 1, binding++)
		{
			if (!(attrBit & vtxInputParams.enabledAttribFlags))
				continue;

			_IRR_DEBUG_BREAK_IF(!(attrBit & outVtxInputParams.enabledAttribFlags));

			const int attrBindingOutMeshBuffer = findAttribIndexBoundToN(binding, outVtxInputParams);
			const int attrBindingCurrMeshBuffer = findAttribIndexBoundToN(binding, vtxInputParams);

			assert((attrBindingOutMeshBuffer != -1) && (attrBindingCurrMeshBuffer != -1)); //imposibru

			if (vtxInputParams.attributes[attrBindingCurrMeshBuffer].format != outVtxInputParams.attributes[attrBindingOutMeshBuffer].format)
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

	const size_t vtxSize = meshBuffers[0]->calcVertexSize();

	if (idxCnt > std::numeric_limits<uint16_t>::max())
	{
		//TODO: create multiple draw calls in this case
		//TODO: if any mesh buffer has E_INDEX_TYPE::EIT_32BIT, then make sure that none of its indices is > std::numeric_limits<uint16_t>::max()
		_IRR_DEBUG_BREAK_IF(true);
		return {};
	}

	//how is bad alloc handled? is there custom new_handler implemented or something?
	//TODO: alignment
	//SBufferBinding<ICPUBuffer> packedVtxBuff{ 0u, core::make_smart_refctd_ptr<ICPUBuffer*>(vtxSize * vtxCnt) };
	SBufferBinding<ICPUBuffer> packedIdxBuff{ 0u, core::make_smart_refctd_ptr<ICPUBuffer*>(sizeof(uint16_t) * idxCnt) };
	constexpr E_INDEX_TYPE packedMeshIndexType = E_INDEX_TYPE::EIT_16BIT;


	return {};
}

}
}