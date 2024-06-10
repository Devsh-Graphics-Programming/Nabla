#include <nbl/builtin/hlsl/cpp_compat.hlsl>

#ifdef __HLSL_VERSION
#define LERP lerp
#else
#define LERP nbl::hlsl::lerp
#endif

#ifdef __HLSL_VERSION
#define ABS abs
#else
#define ABS std::abs
#endif

// TODO: inline function
#define EXCHANGE(a, b) \
   do {                \
       a ^= b;         \
       b ^= a;         \
       a ^= b;         \
   } while (false)
   
#define FLOAT_ROUND_NEAREST_EVEN    0
#define FLOAT_ROUND_TO_ZERO         1
#define FLOAT_ROUND_DOWN            2
#define FLOAT_ROUND_UP              3
#define FLOAT_ROUNDING_MODE         FLOAT_ROUND_NEAREST_EVEN

namespace emulated
{
    namespace impl
    {
        nbl::hlsl::uint32_t2 umulExtended(uint32_t lhs, uint32_t rhs)
        {
            uint64_t product = uint64_t(lhs) * uint64_t(rhs);
            nbl::hlsl::uint32_t2 output;
            output.x = uint32_t((product & 0xFFFFFFFF00000000) >> 32);
            output.y = uint32_t(product & 0x00000000FFFFFFFFull);
            return output;
        }
        
        bool isNaN64(uint64_t val)
        {
            return bool((0x7FF0000000000000ull & val) && (0x000FFFFFFFFFFFFFull & val));
        }

        nbl::hlsl::uint32_t2 add64(uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1)
        {
           nbl::hlsl::uint32_t2 output;
           output.y = a1 + b1;
           output.x = a0 + b0 + uint32_t(output.y < a1);

           return output;
        }


        nbl::hlsl::uint32_t2 sub64(uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1)
        {
            nbl::hlsl::uint32_t2 output;
            output.y = a1 - b1;
            output.x = a0 - b0 - uint32_t(a1 < b1);
            
            return output;
        }

        // TODO: test
        int countLeadingZeros32(uint32_t val)
        {
#ifndef __HLSL_VERSION
            return 31 - nbl::hlsl::findMSB(val);
#else
            return 31 - firstbithigh(val);
#endif
        }
        
        uint64_t propagateFloat64NaN(uint64_t a, uint64_t b)
        {
        #if defined RELAXED_NAN_PROPAGATION
            return a | b;
        #else
        
            bool aIsNaN = isNaN64(a);
            bool bIsNaN = isNaN64(b);
            a |= 0x0008000000000000ull;
            b |= 0x0008000000000000ull;
        
            // TODO:
            //return LERP(b, LERP(a, b, nbl::hlsl::float32_t2(bIsNaN, bIsNaN)), nbl::hlsl::float32_t2(aIsNaN, aIsNaN));
            return 0xdeadbeefbadcaffeull;
        #endif
        }



        nbl::hlsl::uint32_t2 shortShift64Left(uint32_t a0, uint32_t a1, int count)
        {
            nbl::hlsl::uint32_t2 output;
            output.y = a1 << count;
            output.x = LERP((a0 << count | (a1 >> ((-count) & 31))), a0, count == 0);
            
            return output;
        };
        
        nbl::hlsl::uint32_t2 shift64RightJamming(uint32_t a0, uint32_t a1, int count)
        {
            nbl::hlsl::uint32_t2 output;
            const int negCount = (-count) & 31;
        
            output.x = LERP(0u, a0, count == 0);
            output.x = LERP(output.x, (a0 >> count), count < 32);
        
            output.y = uint32_t((a0 | a1) != 0u); /* count >= 64 */
            uint32_t z1_lt64 = (a0>>(count & 31)) | uint32_t(((a0<<negCount) | a1) != 0u);
            output.y = LERP(output.y, z1_lt64, count < 64);
            output.y = LERP(output.y, (a0 | uint32_t(a1 != 0u)), count == 32);
            uint32_t z1_lt32 = (a0<<negCount) | (a1>>count) | uint32_t ((a1<<negCount) != 0u);
            output.y = LERP(output.y, z1_lt32, count < 32);
            output.y = LERP(output.y, a1, count == 0);
            
            return output;
        }

        
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
        
           output.z = LERP(uint32_t(a0 != 0u), a0, count == 64);
           output.z = LERP(output.z, a0 << negCount, count < 64);
           output.z = LERP(output.z, a1 << negCount, count < 32);
        
           output.y = LERP(0u, (a0 >> (count & 31)), count < 64);
           output.y = LERP(output.y, (a0<<negCount) | (a1>>count), count < 32);
        
           a2 = LERP(a2 | a1, a2, count < 32);
           output.x = LERP(output.x, a0 >> count, count < 32);
           output.z |= uint32_t(a2 != 0u);
        
           output.x = LERP(output.x, 0u, (count == 32));
           output.y = LERP(output.y, a0, (count == 32));
           output.z = LERP(output.z, a1, (count == 32));
           output.x = LERP(output.x, a0, (count == 0));
           output.y = LERP(output.y, a1, (count == 0));
           output.z = LERP(output.z, a2, (count == 0));
           
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
                    increment = (FLOAT_ROUNDING_MODE == FLOAT_ROUND_DOWN) && (zFrac2 != 0u);
                 }
                 else
                 {
                    increment = (FLOAT_ROUNDING_MODE == FLOAT_ROUND_UP) && (zFrac2 != 0u);
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
              zExp = LERP(zExp, 0, (zFrac0 | zFrac1) == 0u);
           }
           
           return packFloat64(zSign, zExp, zFrac0, zFrac1);
        }
        
        uint64_t normalizeRoundAndPackFloat64(uint32_t sign, int exp, uint32_t frac0, uint32_t frac1)
        {
           int shiftCount;
           nbl::hlsl::uint32_t3 frac = nbl::hlsl::uint32_t3(frac0, frac1, 0u);
        
           if (frac.x == 0u)
           {
              exp -= 32;
              frac.x = frac.y;
              frac.y = 0u;
           }
        
           shiftCount = countLeadingZeros32(frac.x) - 11;
           if (0 <= shiftCount)
           {
              frac.xy = shortShift64Left(frac.x, frac.y, shiftCount);
           }
           else
           {
              frac.xyz = shift64ExtraRightJamming(frac.x, frac.y, 0u, -shiftCount);
           }
           exp -= shiftCount;
           return roundAndPackFloat64(sign, exp, frac.x, frac.y, frac.z);
        }
        
        static const uint64_t SIGN_MASK = 0x8000000000000000ull;
        static const uint64_t EXP_MASK = 0x7FF0000000000000ull;
        static const uint64_t MANTISA_MASK = 0x000FFFFFFFFFFFFFull;
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
#ifndef __HLSL_VERSION
            emulated_float64_t output;
            output.data = reinterpret_cast<storage_t&>(val);
            return output;
#else
            uint32_t lowBits;
            uint32_t highBits;
            asuint(val, lowBits, highBits);
            
            emulated_float64_t output;
            output.data = (uint64_t(highBits) << 32) | uint64_t(lowBits);
            return output;
#endif
        }
        
        static emulated_float64_t createEmulatedFloat64PreserveBitPattern(uint64_t val)
        {
            emulated_float64_t output;
            output.data = val;
            return output;
        }
        
        // TODO: won't not work for uints with msb of index > 52
#ifndef __HLSL_VERSION
        template<>
        static emulated_float64_t create(uint64_t val)
        {
#ifndef __HLSL_VERSION
            const uint64_t msbIndex = nbl::hlsl::findMSB(val);
#else
            const uint64_t msbIndex = firstbithigh(val);
#endif
            uint64_t exp = ((msbIndex + 1023) << 52) & 0x7FF0000000000000;
            uint64_t mantissa = (val << (52 - msbIndex)) & 0x000FFFFFFFFFFFFFull;
            emulated_float64_t output;
            output.data = exp | mantissa;
            return output;
        }
#endif        
        // TODO: temporary, remove
#ifndef __HLSL_VERSION
        template<>
        static emulated_float64_t create(double val)
        {
            emulated_float64_t output;
            output.data = reinterpret_cast<uint64_t&>(val);
            return output;
        }
#endif

        // arithmetic operators
        emulated_float64_t operator+(const emulated_float64_t rhs)
        {
            emulated_float64_t retval = createEmulatedFloat64PreserveBitPattern(0u);

            uint32_t lhsSign = uint32_t((data & 0x8000000000000000ull) >> 32);
            uint32_t rhsSign = uint32_t((rhs.data & 0x8000000000000000ull) >> 32);

            uint32_t lhsLow = uint32_t(data & 0x00000000FFFFFFFFull);
            uint32_t rhsLow = uint32_t(rhs.data & 0x00000000FFFFFFFFull);
            uint32_t lhsHigh = uint32_t((data & 0x000FFFFF00000000ull) >> 32);
            uint32_t rhsHigh = uint32_t((rhs.data & 0x000FFFFF00000000ull) >> 32);

            int lhsExp = int((data >> 52) & 0x7FFull);
            int rhsExp = int((rhs.data >> 52) & 0x7FFull);
            
            int expDiff = lhsExp - rhsExp;

            if (lhsSign == rhsSign)
            {
                nbl::hlsl::uint32_t3 frac;
                int exp;

                if (expDiff == 0)
                {
                   //if (lhsExp == 0x7FF)
                   //{
                   //   bool propagate = (lhsMantissa | rhsMantissa) != 0u;
                   //   return createEmulatedFloat64PreserveBitPattern(LERP(data, impl::propagateFloat64NaN(data, rhs.data), propagate));
                   //}
                   
                   frac.xy = impl::add64(lhsHigh, lhsLow, rhsHigh, rhsLow);
                   if (lhsExp == 0)
                      return createEmulatedFloat64PreserveBitPattern(impl::packFloat64(lhsSign, 0, frac.x, frac.y));
                   frac.z = 0u;
                   frac.x |= 0x00200000u;
                   exp = lhsExp;
                   frac = impl::shift64ExtraRightJamming(frac.x, frac.y, frac.z, 1);
                }
                else
                {
                     if (expDiff < 0)
                     {
                        EXCHANGE(lhsHigh, rhsHigh);
                        EXCHANGE(lhsLow, rhsLow);
                        EXCHANGE(lhsExp, rhsExp);
                        EXCHANGE(lhsExp, rhsExp);
                     }

                     if (lhsExp == 0x7FF)
                     {
                        bool propagate = (lhsHigh | lhsLow) != 0u;
                        return createEmulatedFloat64PreserveBitPattern(LERP(0x7FF0000000000000ull | (uint64_t(lhsSign) << 32), impl::propagateFloat64NaN(data, rhs.data), propagate));
                     }

                     expDiff = LERP(ABS(expDiff), ABS(expDiff) - 1, rhsExp == 0);
                     rhsHigh = LERP(rhsHigh | 0x00100000u, rhsHigh, rhsExp == 0);
                     nbl::hlsl::float32_t3 shifted = impl::shift64ExtraRightJamming(rhsHigh, rhsLow, 0u, expDiff);
                     rhsHigh = shifted.x;
                     rhsLow = shifted.y;
                     frac.z = shifted.z;
                     exp = lhsExp;

                     lhsHigh |= 0x00100000u;
                     frac.xy = impl::add64(lhsHigh, lhsLow, rhsHigh, rhsLow);
                     --exp;
                     if (!(frac.x < 0x00200000u))
                     {
                         frac = impl::shift64ExtraRightJamming(frac.x, frac.y, frac.z, 1);
                         ++exp;
                     }
                     
                     return createEmulatedFloat64PreserveBitPattern(impl::roundAndPackFloat64(lhsSign, exp, frac.x, frac.y, frac.z));
                }
                
                // cannot happen but compiler cries about not every path returning value
                return createEmulatedFloat64PreserveBitPattern(0xdeadbeefbadcaffeull);
            }
            else
            {
                int exp;
                
                nbl::hlsl::uint32_t2 lhsShifted = impl::shortShift64Left(lhsHigh, lhsLow, 10);
                lhsHigh = lhsShifted.x;
                lhsLow = lhsShifted.y;
                nbl::hlsl::uint32_t2 rhsShifted = impl::shortShift64Left(rhsHigh, rhsLow, 10);
                rhsHigh = rhsShifted.x;
                rhsLow = rhsShifted.y;
                
                if (expDiff != 0)
                {
                    nbl::hlsl::uint32_t2 frac;
                
                    if (expDiff < 0)
                    {
                       EXCHANGE(lhsHigh, rhsHigh);
                       EXCHANGE(lhsLow, rhsLow);
                       EXCHANGE(lhsExp, rhsExp);
                       lhsSign ^= 0x80000000u;
                    }
                    
                    //if (lhsExp == 0x7FF)
                    //{
                    //   bool propagate = (lhsHigh | lhsLow) != 0u;
                    //   return nbl::hlsl::lerp(__packFloat64(lhsSign, 0x7ff, 0u, 0u), __propagateFloat64NaN(a, b), propagate);
                    //}
                    
                    expDiff = LERP(ABS(expDiff), ABS(expDiff) - 1, rhsExp == 0);
                    rhsHigh = LERP(rhsHigh | 0x40000000u, rhsHigh, rhsExp == 0);
                    nbl::hlsl::uint32_t2 shifted = impl::shift64RightJamming(rhsHigh, rhsLow, expDiff);
                    rhsHigh = shifted.x;
                    rhsLow = shifted.y;
                    lhsHigh |= 0x40000000u;
                    frac.xy = impl::sub64(lhsHigh, lhsLow, rhsHigh, rhsLow);
                    exp = lhsExp;
                    --exp;
                    return createEmulatedFloat64PreserveBitPattern(impl::normalizeRoundAndPackFloat64(lhsSign, exp - 10, frac.x, frac.y));
                }
                //if (lhsExp == 0x7FF)
                //{
                //   bool propagate = ((lhsHigh | rhsHigh) | (lhsLow | rhsLow)) != 0u;
                //   return nbl::hlsl::lerp(0xFFFFFFFFFFFFFFFFUL, __propagateFloat64NaN(a, b), propagate);
                //}
                rhsExp = LERP(rhsExp, 1, lhsExp == 0);
                lhsExp = LERP(lhsExp, 1, lhsExp == 0);
                
                nbl::hlsl::uint32_t2 frac;
                uint32_t signOfDifference = 0;
                if (rhsHigh < lhsHigh)
                {
                   frac.xy = impl::sub64(lhsHigh, lhsLow, rhsHigh, rhsLow);
                }
                else if (lhsHigh < rhsHigh)
                {
                   frac.xy = impl::sub64(rhsHigh, rhsLow, lhsHigh, lhsLow);
                   signOfDifference = 0x80000000;
                }
                else if (rhsLow <= lhsLow)
                {
                   /* It is possible that frac.x and frac.y may be zero after this. */
                   frac.xy = impl::sub64(lhsHigh, lhsLow, rhsHigh, rhsLow);
                }
                else
                {
                   frac.xy = impl::sub64(rhsHigh, rhsLow, lhsHigh, lhsLow);
                   signOfDifference = 0x80000000;
                }
                
                exp = LERP(rhsExp, lhsExp, signOfDifference == 0u);
                lhsSign ^= signOfDifference;
                uint64_t retval_0 = impl::packFloat64(uint32_t(FLOAT_ROUNDING_MODE == FLOAT_ROUND_DOWN) << 31, 0, 0u, 0u);
                uint64_t retval_1 = impl::normalizeRoundAndPackFloat64(lhsSign, exp - 11, frac.x, frac.y);
                return createEmulatedFloat64PreserveBitPattern(LERP(retval_0, retval_1, frac.x != 0u || frac.y != 0u));
            }
        }

        emulated_float64_t operator-(emulated_float64_t rhs)
        {
            emulated_float64_t lhs = createEmulatedFloat64PreserveBitPattern(data);
            emulated_float64_t rhsFlipped = rhs.flipSign();
            
            return lhs + rhsFlipped;
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
            uint64_t sign = uint32_t(((data ^ rhs.data) & 0x8000000000000000ull) >> 32);
            

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
            
            return createEmulatedFloat64PreserveBitPattern(impl::roundAndPackFloat64(sign, exp, fracUnpacked.x, fracUnpacked.y, fracUnpacked.z));
        }

        // TODO
        emulated_float64_t operator/(const emulated_float64_t rhs)
        {
            return createEmulatedFloat64PreserveBitPattern(0xdeadbeefbadcaffeull);
        }

        // relational operators
        bool operator==(const emulated_float64_t rhs) { return !(data ^ rhs.data); }
        bool operator!=(const emulated_float64_t rhs) { return data ^ rhs.data; }
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
        
        // TODO: should modify self?
        emulated_float64_t flipSign()
        {
            const uint64_t flippedSign = ((~data) & 0x8000000000000000ull);
            return createEmulatedFloat64PreserveBitPattern(flippedSign | (data & 0x7FFFFFFFFFFFFFFFull));
        }
        
        bool isNaN()
        {
            return impl::isNaN64(data);
        }
    };
    
    //_NBL_STATIC_INLINE_CONSTEXPR emulated_float64_t EMULATED_FLOAT64_NAN = emulated_float64_t::create(0.0 / 0.0);
}
