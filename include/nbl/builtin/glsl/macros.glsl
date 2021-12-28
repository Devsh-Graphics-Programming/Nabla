#ifndef _NBL_BUILTIN_GLSL_MACROS_INCLUDED_
#define _NBL_BUILTIN_GLSL_MACROS_INCLUDED_

#define NBL_GLSL_EVAL(X) X


#define NBL_GLSL_EQUAL(X,Y) (NBL_GLSL_EVAL(X)==NBL_GLSL_EVAL(Y))
#define NBL_GLSL_NOT_EQUAL(X,Y) (NBL_GLSL_EVAL(X)!=NBL_GLSL_EVAL(Y))

#define NBL_GLSL_LESS(X,Y) (NBL_GLSL_EVAL(X)<NBL_GLSL_EVAL(Y))
#define NBL_GLSL_GREATER(X,Y) (NBL_GLSL_EVAL(X)>NBL_GLSL_EVAL(Y))


#define NBL_GLSL_IS_NOT_POT(v) (NBL_GLSL_EVAL(v)&(NBL_GLSL_EVAL(v)-1))!=0
#define NBL_GLSL_IS_POT(v) (!NBL_GLSL_IS_NOT_POT(v))
/**
#define NBL_GLSL_ROUND_UP_POT(v) (1u + \
(((((((((v) - 1u) | (((v) - 1u) >> 0x10u) | \
      (((v) - 1u) | (((v) - 1u) >> 0x10u) >> 0x08u)) | \
     ((((v) - 1u) | (((v) - 1u) >> 0x10u) | \
      (((v) - 1u) | (((v) - 1u) >> 0x10u) >> 0x08u)) >> 0x04u))) | \
   ((((((v) - 1u) | (((v) - 1u) >> 0x10u) | \
      (((v) - 1u) | (((v) - 1u) >> 0x10u) >> 0x08u)) | \
     ((((v) - 1u) | (((v) - 1u) >> 0x10u) | \
      (((v) - 1u) | (((v) - 1u) >> 0x10u) >> 0x08u)) >> 0x04u))) >> 0x02u))) | \
 ((((((((v) - 1u) | (((v) - 1u) >> 0x10u) | \
      (((v) - 1u) | (((v) - 1u) >> 0x10u) >> 0x08u)) | \
     ((((v) - 1u) | (((v) - 1u) >> 0x10u) | \
      (((v) - 1u) | (((v) - 1u) >> 0x10u) >> 0x08u)) >> 0x04u))) | \
   ((((((v) - 1u) | (((v) - 1u) >> 0x10u) | \
      (((v) - 1u) | (((v) - 1u) >> 0x10u) >> 0x08u)) | \
     ((((v) - 1u) | (((v) - 1u) >> 0x10u) | \
      (((v) - 1u) | (((v) - 1u) >> 0x10u) >> 0x08u)) >> 0x04u))) >> 0x02u))) >> 0x01u))))
**/

#define NBL_GLSL_CONCAT_IMPL2(X,Y) X ## Y
#define NBL_GLSL_CONCAT_IMPL(X,Y) NBL_GLSL_CONCAT_IMPL2(X,Y)
#define NBL_GLSL_CONCATENATE2(X,Y) NBL_GLSL_CONCAT_IMPL(NBL_GLSL_EVAL(X),NBL_GLSL_EVAL(Y))
#define NBL_GLSL_CONCATENATE3(X,Y,Z) NBL_GLSL_CONCATENATE2(NBL_GLSL_CONCATENATE2(X,Y),NBL_GLSL_EVAL(Z))
#define NBL_GLSL_CONCATENATE4(X,Y,Z,W) NBL_GLSL_CONCATENATE2(NBL_GLSL_CONCATENATE2(X,Y),NBL_GLSL_CONCATENATE2(Z,W))

#define NBL_GLSL_AND(X,Y) (NBL_GLSL_EVAL(X)&NBL_GLSL_EVAL(Y))

#define NBL_GLSL_ADD(X,Y) (NBL_GLSL_EVAL(X)+NBL_GLSL_EVAL(Y))
#define NBL_GLSL_SUB(X,Y) (NBL_GLSL_EVAL(X)-NBL_GLSL_EVAL(Y))

// https://github.com/google/shaderc/issues/1155
//#define NBL_GLSL_MAX(X,Y) (((NBL_GLSL_EVAL(X))>(NBL_GLSL_EVAL(Y))) ? (NBL_GLSL_EVAL(X)):(NBL_GLSL_EVAL(Y)))
//#define NBL_GLSL_MIN(X,Y) (((NBL_GLSL_EVAL(X))<(NBL_GLSL_EVAL(Y))) ? (NBL_GLSL_EVAL(X)):(NBL_GLSL_EVAL(Y)))

#endif
