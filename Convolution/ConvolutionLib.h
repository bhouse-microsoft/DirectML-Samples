#pragma once

#include <stdint.h>

template<typename T> struct convolution_buffers
{
    std::vector<T> m_input;
    std::vector<T> m_filter;
    std::vector<T> m_output;
};

struct convolution_shape {
    uint32_t batchCount;
    uint32_t channelCount;
    uint32_t inputSize[2];
    uint32_t featureCount;
    uint32_t kernelSize[2];
    uint32_t padding[2];
    uint32_t kernelStride[2];
};

struct convolution_constants {
    uint32_t inputSize[4];
    uint32_t inputStride[4];
    uint32_t inputTotalSize;

    uint32_t filterSize[4];
    uint32_t filterStride[4];
    uint32_t filterTotalSize;

    uint32_t outputSize[4];
    uint32_t outputStride[4];
    uint32_t outputTotalSize;

    uint32_t kernelStride[2];
    uint32_t startPadding[2];

    convolution_constants(convolution_shape & shape)
    {
        inputSize[0] = shape.batchCount;
        inputSize[1] = shape.channelCount;
        for(int i = 0; i < 2; i++)
            inputSize[i+2] = shape.inputSize[i];

        inputStride[3] = 1;
        for(int i = 2; i >= 0; i--)
            inputStride[i] = inputSize[i+1] * inputStride[i+1];

        inputTotalSize = inputSize[0] * inputStride[0];

        filterSize[0] = shape.featureCount;
        filterSize[1] = shape.channelCount;
        for (int i = 0; i < 2; i++)
            filterSize[i + 2] = filterSize[i];

        filterStride[3] = 1;
        for (int i = 2; i >= 0; i--)
            filterStride[i] = filterSize[i + 1] * filterStride[i + 1];

        filterTotalSize = filterSize[0] * filterStride[0];

        outputSize[0] = shape.batchCount;
        outputSize[1] = shape.featureCount;
        for(int i = 0; i < 2; i++)
            outputSize[2+i] = ((shape.inputSize[i] + shape.padding[i]) - ((shape.kernelSize[i] - 1) / 2)) / shape.kernelStride[i];

        outputStride[3] = 1;
        for (int i = 2; i >= 0; i--)
            outputStride[i] = outputSize[i + 1] * outputStride[i + 1];


    }

};

template<typename T>
struct convolution_parameters {
    convolution_constants m_constants;
    convolution_buffers<T> m_buffers;
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
    return dot4(cp.m_constants.filterStrides, f);
}

template<typename T>
T get_filter_value(convolution_parameters<T> & cp, uint32_t * f)
{

    return cp.m_buffers.m_filter[get_filter_offset(cp, f)];
}

template<typename T>
uint32_t get_output_offset(convolution_parameters<T> & cp, uint32_t *o)
{
    return dot4(cp.m_constants.outputStrides, o);
}

template<typename T>
bool get_input_offset(convolution_parameters<T> & cp, uint32_t * f, uint32_t *o, uint32_t *offset)
{
    uint32_t i[4];
    i[0] = 0;
    i[1] = f[1];

    for (int j = 0; j < 2; j++) {
        uint32_t offset = (o[j + 2] * cp.m_constants.kernelStrides[j]) + (f[j + 2]);

        if (offset < cp.m_constants.startPadding[j]) return false;
        offset -= cp.m_constants.startPadding[j];

        if (offset >= cp.m_constants.inputSizes[j + 2]) return false;
        i[j + 2] = offset;
    }

    *offset = dot4(cp.m_constants.inputStrides, i);
    return true;
}

template<typename T>
T get_input_value(convolution_parameters<T> & cp, uint32_t * f, uint32_t *o)
{
    uint32_t offset;
    if (!get_input_offset(cp, f, o, &offset)) return 0;
    return cp.m_buffers.m_input[offset];
}

template<typename T>
float kernel(convolution_parameters<T> & cp, uint32_t * o)
{
    T sum = 0.0;

    uint32_t f[4];
    f[0] = o[1];
    for (f[1] = 0; f[1] < cp.m_constants.filterSizes[1]; f[1]++)
        for (f[2] = 0; f[2] < cp.m_constants.filterSizes[2]; f[2]++)
            for (f[3] = 0; f[3] < cp.m_constants.filterSizes[3]; f[3]++)
                sum += get_filter_value(cp, f) * get_input_value(cp, f, o);

    return sum;
}

template<typename T>
void convolution_evaluate(convolution_parameters<T> & cp)
{
    convolution_constants & c = cp.m_constants;
    assert(c.inputSize[0] == 1);
    assert(c.inputStride[3] == 1);
    assert(c.outputSize[0] == 1);
    assert(c.outputStride[3] == 1);
    assert(c.outputStride[2] == c.outputSize[3]);
    assert(c.outputStride[1] == (c.outputSize[3] * c.outputSize[2]));

    T * output = cp.m_buffers.m_output.data();
    uint32_t o[4];
    o[0] = 0;
    for (o[1] = 0; o[1] < c.outputSizes[1]; o[1]++) // c
    {
        for (o[2] = 0; o[2] < c.outputSizes[2]; o[2]++) // y
        {
            for (o[3] = 0; o[3] < c.outputSizes[3]; o[3]++) // x
            {
                *output++ = kernel(cp, o);
            }
        }
    }
}
