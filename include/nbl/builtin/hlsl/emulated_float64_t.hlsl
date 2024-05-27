#include <nbl/builtin/hlsl/cpp_compat.hlsl>

using float32_t = float;
//using emulated_float64_t = double;

namespace emulated
{
    namespace impl
    {
        nbl::hlsl::uint32_t2 umulExtended(uint32_t lhs, uint32_t rhs)
        {
            uint64_t product = uint64_t(lhs) * uint64_t(rhs);
            nbl::hlsl::uint32_t2 output;
            output.x = (product & 0xFFFFFFFF00000000) >> 32;
            output.y = product & 0x00000000FFFFFFFFull;
            return output;
        }

        nbl::hlsl::uint32_t2 add64(uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1)
        {
           nbl::hlsl::uint32_t2 output;
           output.y = a1 + b1;
           output.x = a0 + b0 + uint32_t(output.y < a1);

           return output;
        }

        nbl::hlsl::uint32_t2 shortShift64Left(uint32_t a0, uint32_t a1, int count)
        {
            nbl::hlsl::uint32_t2 output;
            output.y = a1 << count;
            output.x = nbl::hlsl::lerp((a0 << count | (a1 >> ((-count) & 31))), a0, count == 0);
            
            return output;
        };
        
        nbl::hlsl::uint32_t4 mul64to128(uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1)
        {
            uint32_t z0 = 0u;
            uint32_t z1 = 0u;
            uint32_t z2 = 0u;
            uint32_t z3 = 0u;
            uint32_t more1 = 0u;
            uint32_t more2 = 0u;
        
            nbl::hlsl::uint32_t2 z2z3 = umulExtended(a0, b1);
            z2 = z2z3.x;
            z3 = z2z3.y;
            nbl::hlsl::uint32_t2 z1more2 = umulExtended(a1, b0);
            z1 = z1more2.x;
            more2 = z1more2.y;
            nbl::hlsl::uint32_t2 z1z2 = add64(z1, more2, 0u, z2);
            z1 = z1z2.x;
            z2 = z1z2.y;
            nbl::hlsl::uint32_t2 z0more1 = umulExtended(a0, b0);
            z0 = z0more1.x;
            more1 = z0more1.y;
            nbl::hlsl::uint32_t2 z0z1 = add64(z0, more1, 0u, z1);
            z0 = z0z1.x;
            z1 = z0z1.y;
            nbl::hlsl::uint32_t2 more1more2 = umulExtended(a0, b1);
            more1 = more1more2.x;
            more2 = more1more2.y;
            nbl::hlsl::uint32_t2 more1z2 = add64(more1, more2, 0u, z2);
            more1 = more1z2.x;
            z2 = more1z2.y;
            nbl::hlsl::uint32_t2 z0z12 = add64(z0, z1, 0u, more1);
            z0 = z0z12.x;
            z1 = z0z12.y;


            nbl::hlsl::uint32_t4 output;
            output.x = z0;
            output.y = z1;
            output.z = z2;
            output.w = z3;
            return output;
        }
        
        nbl::hlsl::uint32_t3 shift64ExtraRightJamming(uint32_t a0, uint32_t a1, uint32_t a2, int count)
        {
           nbl::hlsl::uint32_t3 output;
           output.x = 0u;
           
           int negCount = (-count) & 31;
        
           output.z = nbl::hlsl::lerp(uint32_t(a0 != 0u), a0, count == 64);
           output.z = nbl::hlsl::lerp(output.z, a0 << negCount, count < 64);
           output.z = nbl::hlsl::lerp(output.z, a1 << negCount, count < 32);
        
           output.y = nbl::hlsl::lerp(0u, (a0 >> (count & 31)), count < 64);
           output.y = nbl::hlsl::lerp(output.y, (a0<<negCount) | (a1>>count), count < 32);
        
           a2 = nbl::hlsl::lerp(a2 | a1, a2, count < 32);
           output.x = nbl::hlsl::lerp(output.x, a0 >> count, count < 32);
           output.z |= uint32_t(a2 != 0u);
        
           output.x = nbl::hlsl::lerp(output.x, 0u, (count == 32));
           output.y = nbl::hlsl::lerp(output.y, a0, (count == 32));
           output.z = nbl::hlsl::lerp(output.z, a1, (count == 32));
           output.x = nbl::hlsl::lerp(output.x, a0, (count == 0));
           output.y = nbl::hlsl::lerp(output.y, a1, (count == 0));
           output.z = nbl::hlsl::lerp(output.z, a2, (count == 0));
           
           return output;
        }
        
        
        uint64_t packFloat64(uint32_t zSign, int zExp, uint32_t zFrac0, uint32_t zFrac1)
        {
           nbl::hlsl::uint32_t2 z;
        
           z.x = zSign + (uint32_t(zExp) << 20) + zFrac0;
           z.y = zFrac1;
           
           uint64_t output = 0u;
           output |= (uint64_t(z.x) << 32) & 0xFFFFFFFF00000000ull;
           output |= uint64_t(z.y);
           return  output;
        }

        
        uint64_t roundAndPackFloat64(uint32_t zSign, int zExp, uint32_t zFrac0, uint32_t zFrac1, uint32_t zFrac2)
        {
           bool roundNearestEven;
           bool increment;
        
           roundNearestEven = true;
           increment = int(zFrac2) < 0;
           if (!roundNearestEven) 
           {
              if (false) //(FLOAT_ROUNDING_MODE == FLOAT_ROUND_TO_ZERO)
              {
                 increment = false;
              } 
              else
              {
                 if (false) //(zSign != 0u)
                 {
                    //increment = (FLOAT_ROUNDING_MODE == FLOAT_ROUND_DOWN) &&
                    //   (zFrac2 != 0u);
                 }
                 else
                 {
                    //increment = (FLOAT_ROUNDING_MODE == FLOAT_ROUND_UP) &&
                    //   (zFrac2 != 0u);
                 }
              }
           }
           if (0x7FD <= zExp)
           {
              if ((0x7FD < zExp) || ((zExp == 0x7FD) && (0x001FFFFFu == zFrac0 && 0xFFFFFFFFu == zFrac1) && increment))
              {
                 if (false) // ((FLOAT_ROUNDING_MODE == FLOAT_ROUND_TO_ZERO) ||
                    // ((zSign != 0u) && (FLOAT_ROUNDING_MODE == FLOAT_ROUND_UP)) ||
                    // ((zSign == 0u) && (FLOAT_ROUNDING_MODE == FLOAT_ROUND_DOWN)))
                 {
                    return packFloat64(zSign, 0x7FE, 0x000FFFFFu, 0xFFFFFFFFu);
                 }
                 
                 return packFloat64(zSign, 0x7FF, 0u, 0u);
              }
           }
        
           if (zExp < 0)
           {
              nbl::hlsl::uint32_t3 shifted = shift64ExtraRightJamming(zFrac0, zFrac1, zFrac2, -zExp);
              zFrac0 = shifted.x;
              zFrac1 = shifted.y;
              zFrac2 = shifted.z;
              zExp = 0;
              
              if (roundNearestEven)
              {
                 increment = zFrac2 < 0u;
              }
              else
              {
                 if (zSign != 0u)
                 {
                    //increment = (FLOAT_ROUNDING_MODE == FLOAT_ROUND_DOWN) && (zFrac2 != 0u);
                 }
                 else
                 {
                    //increment = (FLOAT_ROUNDING_MODE == FLOAT_ROUND_UP) && (zFrac2 != 0u);
                 }
              }
           }
        
           if (increment)
           {
              nbl::hlsl::uint32_t2 added = add64(zFrac0, zFrac1, 0u, 1u);
              zFrac0 = added.x;
              zFrac1 = added.y;
              zFrac1 &= ~((zFrac2 + uint32_t(zFrac2 == 0u)) & uint32_t(roundNearestEven));
           }
           else
           {
              zExp = nbl::hlsl::lerp(zExp, 0, (zFrac0 | zFrac1) == 0u);
           }
           
           return packFloat64(zSign, zExp, zFrac0, zFrac1);
        }


    }

    struct emulated_float64_t
    {
        using storage_t = uint64_t;

        storage_t data;

        // constructors
        // TODO: specializations
        template <typename T>
        static emulated_float64_t create(T val)
        { 
            emulated_float64_t output;
            output.data = val;
            return output;
        }
        
        template<>
        static emulated_float64_t create(double val)
        {
            emulated_float64_t output;
            output.data = reinterpret_cast<uint64_t&>(val);
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
            emulated_float64_t retval = emulated_float64_t::create(0u);
            
            
            uint32_t lhsLow = uint32_t(data & 0x00000000FFFFFFFFull);
            uint32_t rhsLow = uint32_t(rhs.data & 0x00000000FFFFFFFFull);
            uint32_t lhsHigh = uint32_t((data & 0x000FFFFF00000000ull) >> 32);
            uint32_t rhsHigh = uint32_t((rhs.data & 0x000FFFFF00000000ull) >> 32);
            uint32_t lhsExp = uint32_t((data >> 52) & 0x7FFull);
            uint32_t rhsExp = uint32_t((rhs.data >> 52) & 0x7FFull);

            int32_t exp = int32_t(lhsExp + rhsExp) - 0x400u;
            uint64_t sign = (data ^ rhs.data) & 0x8000000000000000ull;
            

            lhsHigh |= 0x00100000u;
            nbl::hlsl::uint32_t2 shifted = emulated::impl::shortShift64Left(rhsHigh, rhsLow, 12);
            rhsHigh = shifted.x;
            rhsLow = shifted.y;

            nbl::hlsl::uint32_t4 fracUnpacked = impl::mul64to128(lhsHigh, lhsLow, rhsHigh, rhsLow);
            fracUnpacked.xy = emulated::impl::add64(fracUnpacked.x, fracUnpacked.y, lhsHigh, lhsLow);
            fracUnpacked.z |= uint32_t(fracUnpacked.w != 0u);
            if (0x00200000u <= fracUnpacked.x)
            {
               fracUnpacked = nbl::hlsl::uint32_t4(impl::shift64ExtraRightJamming(fracUnpacked.x, fracUnpacked.y, fracUnpacked.z, 1), 0u);
               ++exp;
            }
            
            return emulated_float64_t::create(impl::roundAndPackFloat64(sign, exp, fracUnpacked.x, fracUnpacked.y, fracUnpacked.z));
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
        bool operator<=(const emulated_float64_t rhs) { return data <= rhs.data; }
        bool operator>=(const emulated_float64_t rhs) { return data >= rhs.data; }

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
        //operator min16int() { return min16int(data);}
        //operator float64_t() { return float64_t(data); }
        //operator half() { return half(data); }

        //explicit operator int() const { return int(data); }

        // OMITED OPERATORS
        //  - not implementing bitwise and modulo operators since floating point types doesn't support them
        //  - compound operator overload not supported in HLSL
        //  - access operators (dereference and addressof) not supported in HLSL
    };
}
