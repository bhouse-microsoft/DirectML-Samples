#pragma once

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