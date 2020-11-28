#ifndef _IRR_MATH_TYPELESS_ARITHMETIC_INCLUDED_
#define _IRR_MATH_TYPELESS_ARITHMETIC_INCLUDED_

// TODO: change header name to binary_operator_functions.glsl

uint irr_glsl_identityFunction(in uint x) {return x;}

uint irr_glsl_and(in uint x, in uint y) {return x&y;}

uint irr_glsl_xor(in uint x, in uint y) {return x^y;}

uint irr_glsl_or(in uint x, in uint y) {return x|y;}


uint irr_glsl_add(in uint x, in uint y) {return x+y;}
float irr_glsl_add(in float x, in float y) {return x+y;}

uint irr_glsl_mul(in uint x, in uint y) {return x*y;}
float irr_glsl_mul(in float x, in float y) {return x*y;}

#endif
