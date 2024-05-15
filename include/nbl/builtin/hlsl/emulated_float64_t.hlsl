
using float32_t = float;
//using emulated_float64_t = double;

namespace emulated
{
    struct emulated_float64_t
    {
        // TODO: change to `uint64_t` when on the emulation stage
        using storage_t = float32_t;

        storage_t data;

        // constructors
        // TODO: specializations?
        template <typename T>
        static emulated_float64_t create(T val)
        { 
            emulated_float64_t output;
            output.data = val;
            return output;
        }

        // arithmetic operators
        emulated_float64_t operator+(const emulated_float64_t rhs)
        {
            emulated_float64_t retval;
            retval.data = data + rhs.data;
            return retval;
        }

        emulated_float64_t operator-(const emulated_float64_t rhs)
        {
            emulated_float64_t retval;
            retval.data = data - rhs.data;
            return retval;
        }

        emulated_float64_t operator*(const emulated_float64_t rhs)
        {
            emulated_float64_t retval;
            retval.data = data * rhs.data;
            return retval;
        }

        emulated_float64_t operator/(const emulated_float64_t rhs)
        {
            emulated_float64_t retval;
            retval.data = data / rhs.data;
            return retval;
        }

        // relational operators
        bool operator==(const emulated_float64_t rhs) { return !(uint64_t(data) ^ uint64_t(rhs.data)); }
        bool operator!=(const emulated_float64_t rhs) { return uint64_t(data) ^ uint64_t(rhs.data); }
        bool operator<(const emulated_float64_t rhs) { return data < rhs.data; }
        bool operator>(const emulated_float64_t rhs) { return data > rhs.data; }
        bool operator<=(const emulated_float64_t rhs) { return !operator>(rhs); }
        bool operator>=(const emulated_float64_t rhs) { return !operator<(rhs); }

        //logical operators
        bool operator&&(const emulated_float64_t rhs) { return bool(data) && bool(rhs.data); }
        bool operator||(const emulated_float64_t rhs) { return bool(data) || bool(rhs.data); }
        bool operator!() { return !bool(data); }

        // conversion operators
        operator bool() { return bool(data); }
        operator int() { return int(data); }
        operator uint32_t() { return uint32_t(data); }
        operator uint64_t() { return uint64_t(data); }
        operator float() { return float(data); }
#ifdef __HLSL_VERSION
        operator min16int() { return min16int(data);}
        operator float64_t() { return float64_t(data); }
        operator half() { return half(data); }
#else
        operator uint16_t() { return uint16_t(data);}
        operator double() { return double(data); }
#endif

        //explicit operator int() const { return int(data); }

        // HERE OMITED OPERATORS
        //  - not implementing bitwise and modulo operators since floating point types doesn't support them
        //  - compound operator overload not supported in HLSL
        //  - access operators (dereference and addressof) not supported in HLSL
#ifndef __HLSL_VERSION
        // compound assignment operators
        emulated_float64_t operator+=(emulated_float64_t rhs)
        {
            data = data + rhs.data;
            return create(data);
        }
        
        emulated_float64_t operator-=(emulated_float64_t rhs)
        {
            data = data - rhs.data;
            return create(data);
        }
        
        emulated_float64_t operator*=(emulated_float64_t rhs)
        {
            data = data * rhs.data;
            return create(data);
        }
        
        emulated_float64_t operator/=(emulated_float64_t rhs)
        {
            data = data / rhs.data;
            return create(data);
        }
        
        // access operators
        emulated_float64_t operator*() { return *this; }
        emulated_float64_t* operator&() { return this; }
#endif
    };
}
