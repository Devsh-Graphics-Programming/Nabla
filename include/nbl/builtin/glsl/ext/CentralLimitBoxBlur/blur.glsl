#ifndef _NBL_GLSL_EXT_BLUR_INCLUDED_
#define _NBL_GLSL_EXT_BLUR_INCLUDED_

#ifndef _NBL_GLSL_EXT_BLUR_PASSES_PER_AXIS_
#error "You must define `_NBL_GLSL_EXT_BLUR_PASSES_PER_AXIS_`!"
#endif

#ifndef _NBL_GLSL_EXT_BLUR_AXIS_DIM_
#error "You must define _NBL_GLSL_EXT_BLUR_AXIS_DIM_!"
#endif

#include "nbl/builtin/glsl/workgroup/shared_blur.glsl"

#ifdef _NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_
	#if NBL_GLSL_EVAL(_NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_) < NBL_GLSL_EVAL(_NBL_GLSL_EXT_BLUR_SHARED_SIZE_NEEDED_)
		#error "Not enough shared memory declared"
	#endif
#else
	#define _NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ _NBL_GLSL_EXT_BLUR_SHARED_SIZE_NEEDED_
	#define _NBL_GLSL_SCRATCH_SHARED_DEFINED_ nbl_glsl_ext_Blur_scratchShared
	shared uint _NBL_GLSL_SCRATCH_SHARED_DEFINED_[_NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_];
#endif

#include <nbl/builtin/glsl/workgroup/arithmetic.glsl>
#include <nbl/builtin/glsl/workgroup/ballot.glsl>

#define _NBL_GLSL_EXT_BLUR_ITEMS_PER_THREAD_ (_NBL_GLSL_EXT_BLUR_AXIS_DIM_ + _NBL_GLSL_WORKGROUP_SIZE_ - 1)/_NBL_GLSL_WORKGROUP_SIZE_

// Todo: This spillage calculation is hacky! The lower bound of `_NBL_GLSL_EXT_BLUR_SPILLAGE_LOWER_BOUND_` is just an adhoc
// thing which happens to work for all images until now but it could very well break for the next image.
// This:
// `#define _NBL_GLSL_EXT_BLUR_LOCAL_SPILLAGE_ _NBL_GLSL_EXT_BLUR_ITEMS_PER_THREAD_`
// seems to be a safe lower bound, however, it must be inefficient.
#define _NBL_GLSL_EXT_BLUR_SPILLAGE_LOWER_BOUND_ (_NBL_GLSL_EXT_BLUR_ITEMS_PER_THREAD_/4)

#define _NBL_GLSL_EXT_BLUR_IMPL_LOCAL_SPILLAGE_ ((_NBL_GLSL_WORKGROUP_ARITHMETIC_SHARED_SIZE_NEEDED_-1)/_NBL_GLSL_WORKGROUP_SIZE_+1)

#if _NBL_GLSL_EXT_BLUR_IMPL_LOCAL_SPILLAGE_ > _NBL_GLSL_EXT_BLUR_ITEMS_PER_THREAD_
#define _NBL_GLSL_EXT_BLUR_LOCAL_SPILLAGE_ _NBL_GLSL_EXT_BLUR_ITEMS_PER_THREAD_
#elif _NBL_GLSL_EXT_BLUR_IMPL_LOCAL_SPILLAGE_ < _NBL_GLSL_EXT_BLUR_SPILLAGE_LOWER_BOUND_
#define _NBL_GLSL_EXT_BLUR_LOCAL_SPILLAGE_ _NBL_GLSL_EXT_BLUR_SPILLAGE_LOWER_BOUND_
#else
#define _NBL_GLSL_EXT_BLUR_LOCAL_SPILLAGE_ _NBL_GLSL_EXT_BLUR_IMPL_LOCAL_SPILLAGE_
#endif

#ifndef _NBL_GLSL_EXT_BLUR_SET_DATA_DECLARED_
#define _NBL_GLSL_EXT_BLUR_SET_DATA_DECLARED_
void nbl_glsl_ext_Blur_setData(in uvec3 coordinate, in uint channel, in float val);
#endif

#ifndef _NBL_GLSL_EXT_BLUR_GET_PADDED_DATA_DECLARED_
#define _NBL_GLSL_EXT_BLUR_GET_PADDED_DATA_DECLARED_
float nbl_glsl_ext_Blur_getPaddedData(in uvec3 coordinate, in uint channel);
#endif

#ifndef _NBL_GLSL_EXT_BLUR_GET_PARAMETERS_DEFINED_
#error "You need to define `nbl_glsl_ext_Blur_getParameters` and mark `_NBL_GLSL_EXT_BLUR_GET_PARAMETERS_DEFINED_`!"
#endif
#ifndef _NBL_GLSL_EXT_BLUR_SET_DATA_DEFINED_
#error "You need to define `nbl_glsl_ext_Blur_setData` and mark `_NBL_GLSL_EXT_BLUR_SET_DATA_DEFINED_`!"
#endif
#ifndef _NBL_GLSL_EXT_BLUR_GET_PADDED_DATA_DEFINED_
#error "You need to define `nbl_glsl_ext_Blur_getPaddedData` and mark `_NBL_GLSL_EXT_BLUR_GET_PADDED_DATA_DEFINED_`!"
#endif

#define _NBL_GLSL_EXT_BLUR_WRAP_MODE_REPEAT_			0
#define _NBL_GLSL_EXT_BLUR_WRAP_MODE_CLAMP_TO_EDGE_		1
#define _NBL_GLSL_EXT_BLUR_WRAP_MODE_CLAMP_TO_BORDER_	2
#define _NBL_GLSL_EXT_BLUR_WRAP_MODE_MIRROR_			3

#define _NBL_GLSL_EXT_BLUR_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK_	0
#define _NBL_GLSL_EXT_BLUR_BORDER_COLOR_INT_TRANSPARENT_BLACK_		1
#define _NBL_GLSL_EXT_BLUR_BORDER_COLOR_FLOAT_OPAQUE_BLACK_			2
#define _NBL_GLSL_EXT_BLUR_BORDER_COLOR_INT_OPAQUE_BLACK_			3
#define _NBL_GLSL_EXT_BLUR_BORDER_COLOR_FLOAT_OPAQUE_WHITE_			4
#define _NBL_GLSL_EXT_BLUR_BORDER_COLOR_INT_OPAQUE_WHITE_			5

vec4 nbl_glsl_ext_Blur_getBorderColor()
{
	vec4 borderColor = vec4(1.f, 0.f, 1.f, 1.f);
	switch (nbl_glsl_ext_Blur_Parameters_t_getBorderColor())
	{
	case _NBL_GLSL_EXT_BLUR_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK_:
	case _NBL_GLSL_EXT_BLUR_BORDER_COLOR_INT_TRANSPARENT_BLACK_:
		borderColor = vec4(0.f, 0.f, 0.f, 0.f);
		break;

	case _NBL_GLSL_EXT_BLUR_BORDER_COLOR_FLOAT_OPAQUE_BLACK_:
	case _NBL_GLSL_EXT_BLUR_BORDER_COLOR_INT_OPAQUE_BLACK_:
		borderColor = vec4(0.f, 0.f, 0.f, 1.f);
		break;

	case _NBL_GLSL_EXT_BLUR_BORDER_COLOR_FLOAT_OPAQUE_WHITE_:
	case _NBL_GLSL_EXT_BLUR_BORDER_COLOR_INT_OPAQUE_WHITE_:
		borderColor = vec4(1.f, 1.f, 1.f, 1.f);
		break;
	}
	return borderColor;
}

uvec3 nbl_glsl_ext_Blur_getCoordinates(in uint idx)
{
	const uint direction = nbl_glsl_ext_Blur_Parameters_t_getDirection();
	uvec3 result = gl_WorkGroupID;
	result[direction] = (idx * _NBL_GLSL_WORKGROUP_SIZE_) + gl_LocalInvocationIndex;
	return result;
}

#define PREFIX_SUM(idx) (uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[uint(idx)]))

void nbl_glsl_ext_Blur(in uint ch)
{
	float blurred[_NBL_GLSL_EXT_BLUR_ITEMS_PER_THREAD_];
	for (uint i = 0u; i < _NBL_GLSL_EXT_BLUR_ITEMS_PER_THREAD_; ++i)
		blurred[i] = nbl_glsl_ext_Blur_getPaddedData(nbl_glsl_ext_Blur_getCoordinates(i), ch);

	for (uint pass = 0; pass < _NBL_GLSL_EXT_BLUR_PASSES_PER_AXIS_; ++pass)
	{
		float previous_block_sum = 0.f;

		float spill[_NBL_GLSL_EXT_BLUR_LOCAL_SPILLAGE_];
		for (uint i = 0u; i < _NBL_GLSL_EXT_BLUR_LOCAL_SPILLAGE_; ++i)
		{
			spill[i] = nbl_glsl_workgroupInclusiveAdd(blurred[i]) + previous_block_sum;
			previous_block_sum = nbl_glsl_workgroupBroadcast(spill[i], _NBL_GLSL_WORKGROUP_SIZE_ - 1u);
		}

		for (uint i = _NBL_GLSL_EXT_BLUR_LOCAL_SPILLAGE_; i < _NBL_GLSL_EXT_BLUR_ITEMS_PER_THREAD_; ++i)
		{
			float scan_result = nbl_glsl_workgroupInclusiveAdd(blurred[i]) + previous_block_sum;
			previous_block_sum = nbl_glsl_workgroupBroadcast(scan_result, _NBL_GLSL_WORKGROUP_SIZE_ - 1u);

			uint idx = (i * _NBL_GLSL_WORKGROUP_SIZE_) + gl_LocalInvocationIndex;
			_NBL_GLSL_SCRATCH_SHARED_DEFINED_[idx] = floatBitsToUint(scan_result);
		}

		for (uint i = 0u; i < _NBL_GLSL_EXT_BLUR_LOCAL_SPILLAGE_; ++i)
			_NBL_GLSL_SCRATCH_SHARED_DEFINED_[(i * _NBL_GLSL_WORKGROUP_SIZE_) + gl_LocalInvocationIndex] = floatBitsToUint(spill[i]);
		barrier();

		const uint WRAP_MODE = nbl_glsl_ext_Blur_Parameters_t_getWrapMode();

		vec4 borderColor = vec4(1.f, 0.f, 1.f, 1.f);
		if (WRAP_MODE == _NBL_GLSL_EXT_BLUR_WRAP_MODE_CLAMP_TO_BORDER_)
			borderColor = nbl_glsl_ext_Blur_getBorderColor();

		for (uint i = 0; i < _NBL_GLSL_EXT_BLUR_ITEMS_PER_THREAD_; ++i)
		{
			uint idx = (i * _NBL_GLSL_WORKGROUP_SIZE_) + gl_LocalInvocationIndex;

			const int N = _NBL_GLSL_EXT_BLUR_AXIS_DIM_;

			if (idx < N)
			{
				const float radius = nbl_glsl_ext_Blur_Parameters_t_getRadius() * N;
				float left = float(idx) - radius - 1.f;
				float right = float(idx) + radius;
				const int last = N - 1;

				float result = 0.f;

				if (right <= last)
				{
					result = mix(PREFIX_SUM(floor(right)), PREFIX_SUM(ceil(right)), fract(right));
				}
				else
				{
					switch (WRAP_MODE)
					{
						case _NBL_GLSL_EXT_BLUR_WRAP_MODE_CLAMP_TO_EDGE_:
						{
							result = (right - float(last))*(PREFIX_SUM(last) - PREFIX_SUM(last - 1u)) + PREFIX_SUM(last);
						} break;

						case _NBL_GLSL_EXT_BLUR_WRAP_MODE_CLAMP_TO_BORDER_:
						{
							result = PREFIX_SUM(last) + (right - last)*borderColor[ch];
						} break;

						case _NBL_GLSL_EXT_BLUR_WRAP_MODE_REPEAT_:
						{
							const float v_floored = ceil((floor(right) - last) / N) * PREFIX_SUM(last) + PREFIX_SUM(mod(floor(right)-N, N));
							const float v_ceiled = ceil((ceil(right) - last) / N) * PREFIX_SUM(last) + PREFIX_SUM(mod(ceil(right)-N, N));
							result = mix(v_floored, v_ceiled, fract(right));
						} break;

						case _NBL_GLSL_EXT_BLUR_WRAP_MODE_MIRROR_:
						{
							float v_floored;
							{
								const int floored = int(floor(right));
								const int d = floored - last; // distance from the right-most boundary, >=0

								if (mod(d, 2 * N) == N)
								{
									v_floored = ((d + N) / N) * PREFIX_SUM(last);
								}
								else
								{
									const uint period = uint(ceil(float(d)/N));

									if ((period & 0x1u) == 1)
										v_floored = period * PREFIX_SUM(last) + PREFIX_SUM(last) - PREFIX_SUM(last - uint(mod(d, N)));
									else
										v_floored = period * PREFIX_SUM(last) + PREFIX_SUM(mod(d - 1, N));
								}
							}

							float v_ceiled;
							{
								const int ceiled = int(ceil(right));
								const int d = ceiled - last; // distance from the right-most boundary, >=0

								if (mod(d, 2 * N) == N)
								{
									v_ceiled = ((d + N) / N) * PREFIX_SUM(last);
								}
								else
								{
									const uint period = uint(ceil(float(d)/N));

									if ((period & 0x1u) == 1)
										v_ceiled = period * PREFIX_SUM(last) + PREFIX_SUM(last) - PREFIX_SUM(last - uint(mod(d, N)));
									else
										v_ceiled = period * PREFIX_SUM(last) + PREFIX_SUM(mod(d - 1, N));
								}
							}

							result = mix(v_floored, v_ceiled, fract(right));
						} break;
					}
				}

				if (left >= 0)
				{
					result -= mix(PREFIX_SUM(floor(left)), PREFIX_SUM(ceil(left)), fract(left));
				}
				else
				{
					switch (WRAP_MODE)
					{
						case _NBL_GLSL_EXT_BLUR_WRAP_MODE_CLAMP_TO_EDGE_:
						{
							result -= (1.f - abs(left))*PREFIX_SUM(0);
						} break;

						case _NBL_GLSL_EXT_BLUR_WRAP_MODE_CLAMP_TO_BORDER_:
						{
							result -= (left + 1) * (borderColor[ch]);
						} break;

						case _NBL_GLSL_EXT_BLUR_WRAP_MODE_REPEAT_:
						{
							const float v_floored = floor(floor(left) / N) * PREFIX_SUM(last) + PREFIX_SUM(mod(floor(left), N));
							const float v_ceiled = floor(ceil(left) / N) * PREFIX_SUM(last) + PREFIX_SUM(mod(ceil(left), N));
							result -= mix(v_floored, v_ceiled, fract(left));
						} break;

						case _NBL_GLSL_EXT_BLUR_WRAP_MODE_MIRROR_:
						{
							float v_floored;
							{
								const int floored = int(floor(left));
								if (mod(abs(floored + 1), 2 * N) == 0)
								{
									v_floored = -(abs(floored + 1) / N) * PREFIX_SUM(last);
								}
								else
								{
									const uint period = uint(ceil(float(abs(floored + 1)) / N));

									if ((period & 0x1u) == 1)
										v_floored = -1*(period - 1) * PREFIX_SUM(last) - PREFIX_SUM(mod(abs(floored + 1) - 1, N));
									else
										v_floored = -1*(period - 1) * PREFIX_SUM(last) - (PREFIX_SUM(last) - PREFIX_SUM(mod(floored + 1, N) - 1));
								}
							}

							float v_ceiled;
							{
								const int ceiled = int(ceil(left));
								if (ceiled == 0) // Special case, wouldn't be possible for `floored` above
								{
									v_ceiled = 0;
								}
								else if (mod(abs(ceiled + 1), 2 * N) == 0)
								{
									v_ceiled = -(abs(ceiled + 1) / N) * PREFIX_SUM(last);
								}
								else
								{
									const uint period = uint(ceil(float(abs(ceiled + 1)) / N));

									if ((period & 0x1u) == 1)
										v_ceiled = -1*(period - 1) * PREFIX_SUM(last) - PREFIX_SUM(mod(abs(ceiled + 1) - 1, N));
									else
										v_ceiled = -1*(period - 1) * PREFIX_SUM(last) - (PREFIX_SUM(last) - PREFIX_SUM(mod(ceiled + 1, N) - 1));
								}
							}

							result -= mix(v_floored, v_ceiled, fract(left));
						} break;
					}
				}

				blurred[i] = result / (2.f * radius + 1.f);
			}
		}

		for (uint i = 0; i < _NBL_GLSL_EXT_BLUR_ITEMS_PER_THREAD_; ++i)
			nbl_glsl_ext_Blur_setData(nbl_glsl_ext_Blur_getCoordinates(i), ch, blurred[i]);
	}
}

#endif