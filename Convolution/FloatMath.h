#pragma once

#include <stdint.h>

class Float16Compressor
{
    union Bits
    {
        float f;
        int32_t si;
        uint32_t ui;
    };

    static int const shift = 13;
    static int const shiftSign = 16;

    static int32_t const infN = 0x7F800000; // flt32 infinity
    static int32_t const maxN = 0x477FE000; // max flt16 normal as a flt32
    static int32_t const minN = 0x38800000; // min flt16 normal as a flt32
    static int32_t const signN = 0x80000000; // flt32 sign bit

    static int32_t const infC = infN >> shift;
    static int32_t const nanN = (infC + 1) << shift; // minimum flt16 nan as a flt32
    static int32_t const maxC = maxN >> shift;
    static int32_t const minC = minN >> shift;
    static int32_t const signC = signN >> shiftSign; // flt16 sign bit

    static int32_t const mulN = 0x52000000; // (1 << 23) / minN
    static int32_t const mulC = 0x33800000; // minN / (1 << (23 - shift))

    static int32_t const subC = 0x003FF; // max flt32 subnormal down shifted
    static int32_t const norC = 0x00400; // min flt32 normal down shifted

    static int32_t const maxD = infC - maxC - 1;
    static int32_t const minD = minC - subC - 1;

public:

    static uint16_t compress(float value)
    {
        Bits v, s;
        v.f = value;
        uint32_t sign = v.si & signN;
        v.si ^= sign;
        sign >>= shiftSign; // logical shift
        s.si = mulN;
        s.si = (int32_t)(s.f * v.f); // correct subnormals
        v.si ^= (s.si ^ v.si) & -(minN > v.si);
        v.si ^= (infN ^ v.si) & -((infN > v.si) & (v.si > maxN));
        v.si ^= (nanN ^ v.si) & -((nanN > v.si) & (v.si > infN));
        v.ui >>= shift; // logical shift
        v.si ^= ((v.si - maxD) ^ v.si) & -(v.si > maxC);
        v.si ^= ((v.si - minD) ^ v.si) & -(v.si > subC);
        return v.ui | sign;
    }

    static float decompress(uint16_t value)
    {
        Bits v;
        v.ui = value;
        int32_t sign = v.si & signC;
        v.si ^= sign;
        sign <<= shiftSign;
        v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
        v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
        Bits s;
        s.si = mulC;
        s.f *= v.si;
        int32_t mask = -(norC > v.si);
        v.si <<= shift;
        v.si ^= (s.si ^ v.si) & mask;
        v.si |= sign;
        return v.f;
    }
};

template <int precision>
class ufloat
{
public:

    ufloat()
    {
        m_sign = 0;
        m_exponent = 0;
        m_fraction = 0;
    }

    ufloat(float f)
    {
        uint32_t bits = *((uint32_t *)&f);

        m_sign = bits & 0x8000000;
        bits &= 0x7fffffff;

        m_exponent = (bits >> 23) - 127;
        m_fraction = ((uint64_t)(bits & 0x7FFFFF)) << 41;

        Round();
    }

    ufloat(double d)
    {
        uint64_t bits = *((uint64_t *)&d);

        m_sign = bits & 0x8000000000000000ULL;
        bits &= 0x7fffffffffffffffULL;

        m_exponent = (bits >> 52) - 1023;
        m_fraction = ((uint64_t)(bits & 0x7FFFFFFFFFFFFF)) << 12;

        Round();
    }

    double Get()
    {
        uint64_t bits = (m_sign ? 0x8000000000000000ULL : 0);

        bits |= (m_exponent + 1023) << 52;
        bits |= (m_fraction >> 12);

        return *((double *)&bits);
    }

    bool m_sign;
    int64_t  m_exponent;
    uint64_t m_fraction;

private:

    // Round fractional portion to precision using default IEEE rounding mode algorithm (round to nearest tie break to even)
    void Round()
    {
        int bitsToClear = 64 - precision;
        uint64_t clearMask = (1ULL << bitsToClear) - 1;
        uint64_t bits = m_fraction & clearMask;

        m_fraction &= ~clearMask;

        uint64_t roundingMask = (1ULL << (bitsToClear - 1));

        if (bits & roundingMask)
        {
            bits ^= roundingMask;
            if (bits || (m_fraction & (1ULL << bitsToClear))) {
                m_fraction += 1ULL << bitsToClear;
                if (!m_fraction)
                    m_exponent++;
            }
        }
    }
};

double randd();