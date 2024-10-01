#ifndef _NBL_BUILTIN_HLSL_CENTRAL_LIMIT_BLUR_BOX_BLUR_INCLUDED_
#define _NBL_BUILTIN_HLSL_CENTRAL_LIMIT_BLUR_BOX_BLUR_INCLUDED_

#include <nbl/builtin/hlsl/central_limit_blur/common.hlsl>

#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/workgroup/arithmetic.hlsl>
#include <nbl/builtin/hlsl/workgroup/scratch_size.hlsl>

namespace nbl
{
namespace hlsl
{
namespace central_limit_blur
{

template<uint16_t ItemCount>
struct prefix_sum_t {
	using bin_op_t = nbl::hlsl::plus<float32_t>;
	using type_t = bin_op_t::type_t;

	template<class Accessor>
	static type_t __call(NBL_CONST_REF_ARG(type_t) value, NBL_REF_ARG(Accessor) accessor) {
		type_t retval = nbl::hlsl::workgroup::inclusive_scan<bin_op_t, ItemCount>::template __call<Accessor>(value, accessor);
		return retval;
	}
};

// uint32_t floatBitsToUint(float32_t value) {
// 	return uint32_t((value - -1) * float32_t(0xffffffffu) / (1 - -1) + 0.5f);
// }

//TODO: glsl's mod(x, y) = x - y * floor(x/y). Would prolly need to implement a glsl mod
template<class TextrueAccessor, class ScratchAccessor, class SpillAccessor, uint16_t MAX_ITEMS, uint16_t ARITHMETIC_SIZE>
void BoxBlur(
	uint32_t GroupIndex,
	uint16_t PassesPerAxis,
	uint16_t ItemsPerThread, // TODO:  template ?
	uint16_t chIdx,
	const float32_t radius,
	const uint32_t wrapMode,
	const float32_t4 borderColor,
	NBL_REF_ARG(TextrueAccessor) texAccessor,
	NBL_REF_ARG(ScratchAccessor) scratchAccessor,
	NBL_REF_ARG(SpillAccessor) spillAccessor
) {
    float32_t scale = 1.0f / (2 * radius + 1);
	const uint16_t idx = workgroup::SubgroupContiguousIndex(); // SUS

    for (uint16_t pass = 0u; pass < PassesPerAxis; ++pass) {
		float32_t previous_block_sum = 0.f;

		for (uint32_t i = 0u; i < ItemsPerThread; ++i) {
			// SUS
			float32_t sum = prefix_sum_t<MAX_ITEMS>::template __call<SpillAccessor>(texAccessor.get(i, chIdx), spillAccessor);
			nbl::hlsl::glsl::barrier();
			float32_t value = sum + previous_block_sum;
			spillAccessor.set(i, value);
			nbl::hlsl::glsl::barrier();
			// TODO: change to uin32_t?
			previous_block_sum = nbl::hlsl::workgroup::Broadcast<float32_t, SpillAccessor>(value, spillAccessor, glsl::gl_WorkGroupSize().x - 1);
		}

		for (uint32_t i = 0; i < ItemsPerThread; ++i) {
			float32_t sp;
			spillAccessor.get(i, sp);
			scratchAccessor.set(i * glsl::gl_WorkGroupSize().x + GroupIndex, sp);
		}
		nbl::hlsl::glsl::barrier();

		for (uint32_t i = 0; i < ItemsPerThread; ++i) {
			uint32_t scanlineIdx = i * glsl::gl_WorkGroupSize().x + GroupIndex;
			const int32_t last = glsl::gl_WorkGroupSize().x - 1;
			if (scanlineIdx < last) { // SUS +1
				const float32_t r = radius * glsl::gl_WorkGroupSize().x;
				float32_t left = float32_t(scanlineIdx) - r - 1.f;
				float32_t right = float32_t(scanlineIdx) + r;

				float32_t result = 0.f;

				if (ceil(right) <= float32_t(last)) {
					float32_t floorRight;
					float32_t ceilRight;
					scratchAccessor.get(uint16_t(floor(right)), floorRight);
					scratchAccessor.get(uint16_t(ceil(right)), ceilRight);
					result = lerp(floorRight, ceilRight, frac(right));
				} else {
					switch (wrapMode)
					{
						case WRAP_MODE_CLAMP_TO_EDGE:
						{
							float32_t sumLast;
							float32_t sumLastMinusOne;
							scratchAccessor.get(glsl::gl_WorkGroupSize().x - 1, sumLast); // SUS
							scratchAccessor.get(glsl::gl_WorkGroupSize().x - 2, sumLastMinusOne); // SUS
							result = (right - float32_t(last)) * (sumLast - sumLastMinusOne) + sumLast;
						} break;
					}
				}

				if (left >= 0) {
					float32_t floorLeft;
					float32_t ceilLeft;
					scratchAccessor.get(uint16_t(floor(left)), floorLeft);
					scratchAccessor.get(uint16_t(ceil(left)), ceilLeft);
					result = lerp(floorLeft, ceilLeft, frac(left));
				} else {
					switch (wrapMode) {
						case WRAP_MODE_CLAMP_TO_EDGE:
						{
							float32_t val;
							scratchAccessor.get(0u, val);
							result -= (1 - abs(left)) * val;
						} break;
					}
				}

				nbl::hlsl::glsl::barrier();
				texAccessor.set(i, chIdx, result / (2.f * r + 1.f));
			}
		}
	}
}

}
}
}

#endif