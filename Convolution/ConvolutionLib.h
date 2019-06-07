#pragma once

#include <stdint.h>

template<typename T> struct convolution_buffers
{
    std::vector<T> m_input;
    std::vector<T> m_filter;
    std::vector<T> m_output;
};

class convolution_shape {
public:
    uint32_t m_batchCount;
    uint32_t m_channelCount;
    uint32_t m_inputSize[2];
    uint32_t m_featureCount;
    uint32_t m_kernelSize[2];
    uint32_t m_startPadding[2];
    uint32_t m_endPadding[2];
    uint32_t m_kernelStride[2];
};

class convolution_constants {
public:
    uint32_t m_inputSize[4];
    uint32_t m_inputStride[4];
    uint32_t m_inputElementCount;

    uint32_t m_filterSize[4];
    uint32_t m_filterStride[4];
    uint32_t m_filterElementCount;

    uint32_t m_outputSize[4];
    uint32_t m_outputStride[4];
    uint32_t m_outputElementCount;

    uint32_t m_kernelStride[2];
    uint32_t m_startPadding[2];
    uint32_t m_endPadding[2];

    convolution_constants(const convolution_shape & shape)
    {
        assert(shape.m_batchCount > 0);
        assert(shape.m_channelCount > 0);
        assert(shape.m_featureCount > 0);

        m_inputSize[0] = shape.m_batchCount;
        m_inputSize[1] = shape.m_channelCount;
        for (int i = 0; i < 2; i++) {
            assert(shape.m_inputSize[i] > 0);
            m_inputSize[i + 2] = shape.m_inputSize[i];
        }

        m_inputStride[3] = 1;
        for (int i = 2; i >= 0; i--)
            m_inputStride[i] = m_inputSize[i + 1] * m_inputStride[i + 1];

        m_inputElementCount = m_inputSize[0] * m_inputStride[0];
        assert(m_inputElementCount > 0);

        m_filterSize[0] = shape.m_featureCount;
        m_filterSize[1] = shape.m_channelCount;
        for (int i = 0; i < 2; i++) {
            assert(shape.m_kernelSize[i] > 0);
            m_filterSize[i + 2] = shape.m_kernelSize[i];
        }

        m_filterStride[3] = 1;
        for (int i = 2; i >= 0; i--)
            m_filterStride[i] = m_filterSize[i + 1] * m_filterStride[i + 1];

        m_filterElementCount = m_filterSize[0] * m_filterStride[0];
        assert(m_filterElementCount > 0);

        m_outputSize[0] = shape.m_batchCount;
        m_outputSize[1] = shape.m_featureCount;
        for (int i = 0; i < 2; i++) {
            uint32_t inputSize = shape.m_inputSize[i] + shape.m_startPadding[i] + shape.m_endPadding[i];
            assert(inputSize >= shape.m_kernelSize[i]);
            uint32_t inputRemaining = inputSize - shape.m_kernelSize[i];
            assert((inputRemaining % shape.m_kernelStride[i]) == 0);
            uint32_t kernelsRemaining = inputRemaining / shape.m_kernelStride[i];
            m_outputSize[2 + i] = 1 + kernelsRemaining;
        }

        m_outputStride[3] = 1;
        for (int i = 2; i >= 0; i--)
            m_outputStride[i] = m_outputSize[i + 1] * m_outputStride[i + 1];

        m_outputElementCount = m_outputSize[0] * m_outputStride[0];
        assert(m_outputElementCount > 0);

        for (int i = 0; i < 2; i++) {
            m_startPadding[i] = shape.m_startPadding[i];
            m_endPadding[i] = shape.m_endPadding[i];
            assert(shape.m_kernelStride[i] > 0);
            m_kernelStride[i] = shape.m_kernelStride[i];
        }

    }

};

template<typename T>
class convolution_parameters {
public:
    convolution_constants m_constants;
    convolution_buffers<T> m_buffers;

    convolution_parameters(const convolution_shape & shape) : m_constants(shape)
    {
        m_buffers.m_input.resize(m_constants.m_inputElementCount);
        m_buffers.m_filter.resize(m_constants.m_filterElementCount);
        m_buffers.m_output.resize(m_constants.m_outputElementCount);
    }
};

inline uint32_t dot4(uint32_t * a, uint32_t *b)
{
    uint32_t value = 0;
    for (int i = 0; i < 4; i++)
        value += (a[i] * b[i]);

    return value;
}

template<typename T>
uint32_t get_filter_offset(convolution_parameters<T> & cp, uint32_t * f)
{
    return dot4(cp.m_constants.m_filterStride, f);
}

template<typename T>
T get_filter_value(convolution_parameters<T> & cp, uint32_t * f)
{
    uint32_t offset = get_filter_offset(cp, f);
    T value = cp.m_buffers.m_filter[offset];
    return value;
}

template<typename T>
uint32_t get_output_offset(convolution_parameters<T> & cp, uint32_t *o)
{
    return dot4(cp.m_constants.m_outputStride, o);
}

template<typename T>
bool get_input_offset(convolution_parameters<T> & cp, uint32_t * f, uint32_t *o, uint32_t *offset)
{
    uint32_t i[4];
    i[0] = 0;
    i[1] = f[1];

    for (int j = 0; j < 2; j++) {
        uint32_t offset = (o[j + 2] * cp.m_constants.m_kernelStride[j]) + (f[j + 2]);

        if (offset < cp.m_constants.m_startPadding[j]) return false;
        offset -= cp.m_constants.m_startPadding[j];

        if (offset >= cp.m_constants.m_inputSize[j + 2]) return false;
        i[j + 2] = offset;
    }

    *offset = dot4(cp.m_constants.m_inputStride, i);
    return true;
}

template<typename T>
T get_input_value(convolution_parameters<T> & cp, uint32_t * f, uint32_t *o)
{
    uint32_t offset;
    if (!get_input_offset(cp, f, o, &offset)) return 0;
    T value = cp.m_buffers.m_input[offset];
    return value;
}

template<typename T>
T kernel(convolution_parameters<T> & cp, uint32_t * o)
{
    double sum = 0.0;

    uint32_t f[4];
    f[0] = o[1];
    for (f[1] = 0; f[1] < cp.m_constants.m_filterSize[1]; f[1]++)
        for (f[2] = 0; f[2] < cp.m_constants.m_filterSize[2]; f[2]++)
            for (f[3] = 0; f[3] < cp.m_constants.m_filterSize[3]; f[3]++) {
                double product = (double)get_filter_value(cp, f) * (double)get_input_value(cp, f, o);
                float productFloat = (float) abs(product);
                g_maxProduct = (productFloat > g_maxProduct ? productFloat : g_maxProduct);
                g_minProduct = (productFloat < g_minProduct ? productFloat : g_minProduct);
                sum += product;
            }

    return (T) sum;
}

template<typename T>
void convolution_evaluate(convolution_parameters<T> & cp)
{
    convolution_constants & c = cp.m_constants;
    assert(c.m_inputSize[0] == 1);
    assert(c.m_inputStride[3] == 1);
    assert(c.m_outputSize[0] == 1);
    assert(c.m_outputStride[3] == 1);
    assert(c.m_outputStride[2] == c.m_outputSize[3]);
    assert(c.m_outputStride[1] == (c.m_outputSize[3] * c.m_outputSize[2]));

    T * output = cp.m_buffers.m_output.data();
    uint32_t o[4];
    o[0] = 0;
    for (o[1] = 0; o[1] < c.m_outputSize[1]; o[1]++) // c
    {
        for (o[2] = 0; o[2] < c.m_outputSize[2]; o[2]++) // y
        {
            for (o[3] = 0; o[3] < c.m_outputSize[3]; o[3]++) // x
            {
                *output++ = kernel(cp, o);
            }
        }
    }
}
