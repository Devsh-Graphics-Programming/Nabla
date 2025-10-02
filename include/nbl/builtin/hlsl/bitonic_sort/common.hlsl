#ifndef _NBL_BUILTIN_HLSL_BITONIC_SORT_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BITONIC_SORT_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/math/intutil.hlsl>

namespace nbl
{
    namespace hlsl
    {
		namespace bitonic_sort
        {
            template<uint16_t _Log2ElementsPerThread, uint16_t _Log2ThreadsPerSubgroup, uint16_t _Log2SubgroupsPerWorkgroup, typename _KeyType, typename _ValueType = void,
                typename _Scalar NBL_PRIMARY_REQUIRES(_Log2ElementsPerThread > 0 && _Log2ThreadsPerSubgroup >= 4)
            struct ConstevalParameters
            {
				using scalar_t = _Scalar;
                using key_t = _KeyType;
                using value_t = _ValueType;

                struct ThreadConfig
                {
                    NBL_CONSTEXPR_STATIC_INLINE uint16_t ElementsPerThread = uint16_t(1) << _Log2ElementsPerThread;
                };

                struct SubgroupConfig
                {
                    using thread_config_t = ThreadConfig;
                    thread_config_t thread;
                    NBL_CONSTEXPR_STATIC_INLINE uint16_t ThreadsPerSubgroup = uint16_t(1) << _Log2ThreadsPerSubgroup;
                    NBL_CONSTEXPR_STATIC_INLINE uint16_t ElementsPerSubgroup = thread.ElementsPerThread * ThreadsPerSubgroup;
                    NBL_CONSTEXPR_STATIC_INLINE uint16_t Log2ElementsPerSubgroup = thread.Log2ElementsPerThread + Log2ThreadsPerSubgroup;
                };

                struct WorkgroupConfig
                {
                    using subgroup_config_t = SubgroupConfig;
                    subgroup_config_t subgroup;
                    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupsPerWorkgroup = uint16_t(1) << _Log2SubgroupsPerWorkgroup;
                    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = subgroup.ThreadsPerSubgroup * SubgroupsPerWorkgroup; // threads per workgroup
                    NBL_CONSTEXPR_STATIC_INLINE uint32_t ElementsPerWorkgroup = subgroup.ElementsPerSubgroup * SubgroupsPerWorkgroup;
                    NBL_CONSTEXPR_STATIC_INLINE uint16_t Log2ElementsPerWorkgroup = subgroup.Log2ElementsPerSubgroup + Log2SubgroupsPerWorkgroup;
                };

                using thread_config_t = ThreadConfig;
                using subgroup_config_t = SubgroupConfig;
                using workgroup_config_t = WorkgroupConfig;

                workgroup_config_t workgroup;

                NBL_CONSTEXPR_STATIC_INLINE uint32_t TotalSize = uint32_t(1) << workgroup.Log2ElementsPerWorkgroup;

                NBL_CONSTEXPR_STATIC_INLINE uint32_t KeySize = sizeof(key_t);
                NBL_CONSTEXPR_STATIC_INLINE uint32_t ValueSize = sizeof(value_t);  
                NBL_CONSTEXPR_STATIC_INLINE uint32_t SharedMemoryBytes = TotalSize * (KeySize + ValueSize);
                NBL_CONSTEXPR_STATIC_INLINE uint32_t SharedMemoryDWORDs = SharedMemoryBytes / sizeof(uint32_t);

            };

        }
    }
}

#endif