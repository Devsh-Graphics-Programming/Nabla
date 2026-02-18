#ifndef _NBL_SYSTEM_JSON_H_INCLUDED_
#define _NBL_SYSTEM_JSON_H_INCLUDED_

namespace nbl::system::json {
    template<typename T, typename SFINAE = void> struct adl_serializer;
}

#define NBL_JSON_IMPL_BIND_ADL_SERIALIZER(T)        \
namespace nlohmann {                                \
    template<>                                      \
    struct adl_serializer<typename T::value_t>      \
        : T {};                                     \
}

#endif // _NBL_SYSTEM_JSON_H_INCLUDED_