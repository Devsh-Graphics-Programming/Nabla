#ifndef _IRR_MATH_TYPELESS_ARITHMETIC_INCLUDED_
#define _IRR_MATH_TYPELESS_ARITHMETIC_INCLUDED_

uint irr_glsl_identityFunction(in uint x) {return x;}

uint irr_glsl_and(in uint x, in uint y) {return x&y;}

uint irr_glsl_xor(in uint x, in uint y) {return x^y;}

uint irr_glsl_or(in uint x, in uint y) {return x|y;}


uint irr_glsl_add(in uint x, in uint y) {return x+y;}
uint irr_glsl_addAsFloat(in uint x, in uint y) {return floatBitsToUint(uintBitsToFloat(x)+uintBitsToFloat(y));}

uint irr_glsl_mul(in uint x, in uint y) {return x*y;}
uint irr_glsl_mulAsFloat(in uint x, in uint y) {return floatBitsToUint(uintBitsToFloat(x)*uintBitsToFloat(y));}

uint irr_glsl_minAsInt(in uint x, in uint y) {return min(int(x),int(y));}
uint irr_glsl_minAsFloat(in uint x, in uint y) {return floatBitsToUint(min(uintBitsToFloat(x),uintBitsToFloat(y)));}

uint irr_glsl_maxAsInt(in uint x, in uint y) {return max(int(x),int(y));}
uint irr_glsl_maxAsFloat(in uint x, in uint y) {return floatBitsToUint(max(uintBitsToFloat(x),uintBitsToFloat(y)));}

#endif
