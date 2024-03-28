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
struct prefix_sum_t
{
	using bin_op_t = nbl::hlsl::plus<float32_t>;
	using type_t = bin_op_t::type_t;

	template<class Accessor>
	static type_t __call( NBL_CONST_REF_ARG(type_t) value, NBL_REF_ARG(Accessor) accessor )
	{
		type_t retval = nbl::hlsl::workgroup::inclusive_scan<bin_op_t, ItemCount>::template __call<Accessor>( value, accessor );
		return retval;
	}
};

template<class Accessor, uint16_t offset>
struct ArithmeticProxy
{
	Accessor original;
	float32_t get( const uint32_t ix )
	{
		return original.get( ix + offset );
	}
	void set( const uint32_t ix, NBL_CONST_REF_ARG(float32_t) value )
	{
		original.set( ix + offset, value );
	}

	void workgroupExecutionAndMemoryBarrier()
	{
		nbl::hlsl::glsl::barrier();
	}
};


//TODO: glsl's mod(x, y) = x - y * floor(x/y). Would prolly need to implement a glsl mod
template<class TextrueAccessor, class GroupSharedAccessor, class ArithmeticAccessor, uint16_t MAX_ITEMS, uint16_t ARITHMETIC_SIZE>
void BoxBlur(
	uint16_t PassesPerAxis,
	uint16_t ItemsPerThread, // TODO:  template ?
	uint16_t chIdx,
	const float32_t radius,
	const uint32_t wrapMode,
	const float32_t4 borderColor,
	NBL_REF_ARG(TextrueAccessor) texAccessor,
	NBL_REF_ARG(GroupSharedAccessor) sharedAccessor,
	NBL_REF_ARG(ArithmeticAccessor) arithemitcAccessor
) {
	for( uint16_t pass = 0u; pass < PassesPerAxis; ++pass )
	{
		//ArithmeticProxy<GroupSharedAccessor, MAX_ITEMS - 1> arithemitcAccessor = { sharedAccessor };
		const uint16_t idx = workgroup::SubgroupContiguousIndex();
		for( uint16_t i = 0u; i < ItemsPerThread; ++i )
		{
			// broadcast barrier
			arithemitcAccessor.workgroupExecutionAndMemoryBarrier();

			float32_t blurred = texAccessor.get( i, chIdx );
			// add previous to current but only to the first input cause will propagate anyway
			if( idx == 0 )
			{
				blurred += arithemitcAccessor.get( glsl::gl_WorkGroupSize().x - 1u );
			}
			float32_t sum = prefix_sum_t<MAX_ITEMS>::template __call<__decltype( arithemitcAccessor )>( blurred, arithemitcAccessor );
			arithemitcAccessor.workgroupExecutionAndMemoryBarrier();

			//// finish last prefix sum before storing to scratch we overwrite its scratch
			//if( i == ( ITEMS_PER_THREAD - 1 ) )
			//{
			//	arithemitcAccessor.workgroupExecutionAndMemoryBarrier();
			//}
			sharedAccessor.workgroupExecutionAndMemoryBarrier();
			sharedAccessor.set( idx, sum );

			sharedAccessor.workgroupExecutionAndMemoryBarrier();
			
			const uint16_t last = uint16_t( glsl::gl_WorkGroupSize().x );
			const uint16_t scanlineIdx = idx;

			//if( scanlineIdx < last )
			{
				float32_t result = 0.f;

				float32_t left = float32_t( scanlineIdx ) - radius - 1.f;
				float32_t right = float32_t( scanlineIdx ) + radius;
				
				if( ceil( right ) < float32_t( last ) )
				{
					float32_t floorRight = sharedAccessor.get( uint16_t( floor( right ) ) );
					float32_t ceilRight = sharedAccessor.get( uint16_t( ceil( right ) ) );
					result = lerp( floorRight, ceilRight, frac( right ) );
				}
				else
				{
					switch( wrapMode )
					{
					case WRAP_MODE_CLAMP_TO_EDGE: // Only for last ieratrion
					{
						float32_t sumLast = sharedAccessor.get( glsl::gl_WorkGroupSize().x - 1u );
						float32_t sumLastMinusOne = sharedAccessor.get( glsl::gl_WorkGroupSize().x - 2u );
						result = ( right - float32_t( last ) ) * ( sumLast - sumLastMinusOne ) + sumLast;
					} break;
					/*
					case WRAP_MODE_CLAMP_TO_BORDER:
					{
						float32_t sumLast = prefixSumsAccessor.get( last );
						result = sumLast + ( right - last ) * borderColor[ ch ];
					} break;

					case WRAP_MODE_REPEAT:
					{
						float32_t sumLast = prefixSumsAccessor.get( last );
						float32_t sumModFloorRight = prefixSumsAccessor.get( fmod( floor( right ) - AXIS_DIM, AXIS_DIM ) );
						float32_t sumModCeilRight = prefixSumsAccessor.get( fmod( ceil( right ) - AXIS_DIM, AXIS_DIM ) );
						const float32_t v_floored = ceil( ( floor( right ) - last ) / AXIS_DIM ) * sumLast + sumModFloorRight;
						const float32_t v_ceiled = ceil( ( ceil( right ) - last ) / AXIS_DIM ) * sumLast + sumModCeilRight;
						result = lerp( v_floored, v_ceiled, frac( right ) );
					} break;

					case WRAP_MODE_MIRROR:
					{
						float32_t v_floored;
						{
							const uint32_t floored = uint32_t( floor( right ) );
							const uint32_t d = floored - last; // distance from the right-most boundary, >=0

							if( fmod( d, 2 * AXIS_DIM ) == AXIS_DIM )
							{
								v_floored = ( ( d + AXIS_DIM ) / AXIS_DIM ) * prefixSumsAccessor.get( last );
							}
							else
							{
								const uint32_t period = uint32_t( ceil( float32_t( d ) / AXIS_DIM ) );

								if( ( period & 0x1u ) == 1 )
									v_floored = period * prefixSumsAccessor.get( last ) + prefixSumsAccessor.get( last ) - prefixSumsAccessor.get( last - uint32_t( fmod( d, AXIS_DIM ) ) );
								else
									v_floored = period * prefixSumsAccessor.get( last ) + prefixSumsAccessor.get( fmod( d - 1, AXIS_DIM ) );
							}
						}

						float32_t v_ceiled;
						{
							const uint32_t ceiled = uint32_t( ceil( right ) );
							const uint32_t d = ceiled - last; // distance from the right-most boundary, >=0

							if( fmod( d, 2 * AXIS_DIM ) == AXIS_DIM )
							{
								v_ceiled = ( ( d + AXIS_DIM ) / AXIS_DIM ) * prefixSumsAccessor.get( last );
							}
							else
							{
								const uint32_t period = uint32_t( ceil( float32_t( d ) / AXIS_DIM ) );

								if( ( period & 0x1u ) == 1 )
									v_ceiled = period * prefixSumsAccessor.get( last ) + prefixSumsAccessor.get( last ) - prefixSumsAccessor.get( last - uint32_t( fmod( d, AXIS_DIM ) ) );
								else
									v_ceiled = period * prefixSumsAccessor.get( last ) + prefixSumsAccessor.get( fmod( d - 1, AXIS_DIM ) );
							}
						}

						result = lerp( v_floored, v_ceiled, frac( right ) );
					} break;*/
					}
				}
				
				if( left >= 0 )
				{
					float32_t floorLeft = sharedAccessor.get( uint16_t( floor( left ) ) );
					float32_t ceilLeft = sharedAccessor.get( uint16_t( ceil( left ) ) );
					result -= lerp( floorLeft, ceilLeft, frac( left ) );
				}
				else
				{
					switch( wrapMode )
					{
					case WRAP_MODE_CLAMP_TO_EDGE:
					{
						result -= ( 1.f - abs( left ) ) * sharedAccessor.get( 0u );
					} break;
					/*
					case WRAP_MODE_CLAMP_TO_BORDER:
					{
						result -= ( left + 1 ) * ( borderColor[ ch ] );
					} break;

					case WRAP_MODE_REPEAT:
					{
						const float32_t v_floored = floor( floor( left ) / AXIS_DIM ) * prefixSumsAccessor.get( last ) + prefixSumsAccessor.get( fmod( floor( left ), AXIS_DIM ) );
						const float32_t v_ceiled = floor( ceil( left ) / AXIS_DIM ) * prefixSumsAccessor.get( last ) + prefixSumsAccessor.get( fmod( ceil( left ), AXIS_DIM ) );
						result -= lerp( v_floored, v_ceiled, frac( left ) );
					} break;

					case WRAP_MODE_MIRROR:
					{
						float32_t v_floored;
						{
							const uint32_t floored = uint32_t( floor( left ) );
							if( fmod( abs( floored + 1 ), 2 * AXIS_DIM ) == 0 )
							{
								v_floored = -( abs( floored + 1 ) / AXIS_DIM ) * prefixSumsAccessor.get( last );
							}
							else
							{
								const uint32_t period = uint32_t( ceil( float32_t( abs( floored + 1 ) ) / AXIS_DIM ) );

								if( ( period & 0x1u ) == 1 )
									v_floored = -1 * ( period - 1 ) * prefixSumsAccessor.get( last ) - prefixSumsAccessor.get( fmod( abs( floored + 1 ) - 1, AXIS_DIM ) );
								else
									v_floored = -1 * ( period - 1 ) * prefixSumsAccessor.get( last ) - ( prefixSumsAccessor.get( last ) - prefixSumsAccessor.get( fmod( floored + 1, AXIS_DIM ) - 1 ) );
							}
						}

						float32_t v_ceiled;
						{
							const uint32_t ceiled = uint32_t( ceil( left ) );
							if( ceiled == 0 ) // Special case, wouldn't be possible for `floored` above
							{
								v_ceiled = 0;
							}
							else if( fmod( abs( ceiled + 1 ), 2 * AXIS_DIM ) == 0 )
							{
								v_ceiled = -( abs( ceiled + 1 ) / AXIS_DIM ) * prefixSumsAccessor.get( last );
							}
							else
							{
								const uint32_t period = uint32_t( ceil( float32_t( abs( ceiled + 1 ) ) / AXIS_DIM ) );

								if( ( period & 0x1u ) == 1 )
									v_ceiled = -1 * ( period - 1 ) * prefixSumsAccessor.get( last ) - prefixSumsAccessor.get( fmod( abs( ceiled + 1 ) - 1, AXIS_DIM ) );
								else
									v_ceiled = -1 * ( period - 1 ) * prefixSumsAccessor.get( last ) - ( prefixSumsAccessor.get( last ) - prefixSumsAccessor.get( fmod( ceiled + 1, AXIS_DIM ) - 1 ) );
							}
						}

						result -= lerp( v_floored, v_ceiled, frac( left ) );
					} break;*/
					}
				}
				
				float32_t blurredChannel = result / ( 2.f * radius + 1.f );
				nbl::hlsl::glsl::barrier();
				texAccessor.set( i, chIdx, blurredChannel );
			}
		}
	}
}

}
}
}

#endif