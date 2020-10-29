#ifndef _IRR_BUILTIN_GLSL_MACROS_INCLUDED_
#define _IRR_BUILTIN_GLSL_MACROS_INCLUDED_

#define IRR_GLSL_EVAL(X) X

#define IRR_GLSL_IS_POT(v) (v&(v-1u))
/**
#define IRR_GLSL_ROUND_UP_POT(v) (1u + \
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

#endif
