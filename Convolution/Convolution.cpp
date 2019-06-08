// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"

#include "ConvolutionLib.h"
#include "FloatMath.h"

using winrt::com_ptr;
using winrt::check_hresult;
using winrt::check_bool;
using winrt::handle;

class StopWatch
{
public:

    StopWatch()
    {
        QueryPerformanceFrequency(&m_freq);
    }

    void Start()
    {
        QueryPerformanceCounter(&m_start);
    }

    double Stop()
    {
        LARGE_INTEGER end;
        QueryPerformanceCounter(&end);
        return (double)(end.QuadPart - m_start.QuadPart) / (double)m_freq.QuadPart;
    }

private:

    LARGE_INTEGER m_start;
    LARGE_INTEGER m_freq;
};

void InitializeDirect3D12(
    com_ptr<ID3D12Device> & d3D12Device,
    com_ptr<ID3D12CommandQueue> & commandQueue,
    com_ptr<ID3D12CommandAllocator> & commandAllocator,
    com_ptr<ID3D12GraphicsCommandList> & commandList)
{
#if defined(_DEBUG)
    com_ptr<ID3D12Debug> d3D12Debug;
    if (FAILED(D3D12GetDebugInterface(__uuidof(d3D12Debug), d3D12Debug.put_void())))
    {
        // The D3D12 debug layer is missing - you must install the Graphics Tools optional feature
        winrt::throw_hresult(DXGI_ERROR_SDK_COMPONENT_MISSING);
    }
    d3D12Debug->EnableDebugLayer();
#endif

    com_ptr<IDXGIFactory4> dxgiFactory;
    check_hresult(CreateDXGIFactory1(__uuidof(dxgiFactory), dxgiFactory.put_void()));

    com_ptr<IDXGIAdapter> dxgiAdapter;
    UINT adapterIndex{};
    HRESULT hr{};
    do
    {
        dxgiAdapter = nullptr;
        check_hresult(dxgiFactory->EnumAdapters(adapterIndex, dxgiAdapter.put()));
        ++adapterIndex;

        DXGI_ADAPTER_DESC desc;
        dxgiAdapter.get()->GetDesc(&desc);

#if 1
        if (wcsstr(desc.Description, L"Microsoft") == NULL) {
            hr = DXGI_ERROR_UNSUPPORTED;
            continue;
        }
#endif

        hr = ::D3D12CreateDevice(
            dxgiAdapter.get(),
            D3D_FEATURE_LEVEL_12_0,
            __uuidof(d3D12Device),
            d3D12Device.put_void());
        if (hr == DXGI_ERROR_UNSUPPORTED) continue;
        check_hresult(hr);
    } while (hr != S_OK);

    D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
    commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

    check_hresult(d3D12Device->CreateCommandQueue(
        &commandQueueDesc,
        __uuidof(commandQueue),
        commandQueue.put_void()));

    check_hresult(d3D12Device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        __uuidof(commandAllocator),
        commandAllocator.put_void()));

    check_hresult(d3D12Device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        commandAllocator.get(),
        nullptr,
        __uuidof(commandList),
        commandList.put_void()));
}

double CloseExecuteResetWait(
    com_ptr<ID3D12Device> d3D12Device,
    com_ptr<ID3D12CommandQueue> commandQueue,
    com_ptr<ID3D12CommandAllocator> commandAllocator,
    com_ptr<ID3D12GraphicsCommandList> commandList)
{
    check_hresult(commandList->Close());

    com_ptr<ID3D12Fence> d3D12Fence;
    check_hresult(d3D12Device->CreateFence(
        0,
        D3D12_FENCE_FLAG_NONE,
        _uuidof(d3D12Fence),
        d3D12Fence.put_void()));

    handle fenceEventHandle{ 0 };
    fenceEventHandle.attach(::CreateEvent(nullptr, true, false, nullptr));
    check_bool(bool{ fenceEventHandle });

    StopWatch stopWatch;

    stopWatch.Start();
    ID3D12CommandList* commandLists[] = { commandList.get() };
    commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);

    check_hresult(d3D12Fence->SetEventOnCompletion(1, fenceEventHandle.get()));

    check_hresult(commandQueue->Signal(d3D12Fence.get(), 1));
    ::WaitForSingleObjectEx(fenceEventHandle.get(), INFINITE, FALSE);

    double time = stopWatch.Stop();

    check_hresult(commandList->Reset(commandAllocator.get(), nullptr));

    return time;
}

// ===================================================================================================================
//   DML utilities
// ===================================================================================================================

inline UINT64 DMLCalcBufferTensorSize(
    DML_TENSOR_DATA_TYPE dataType,
    UINT dimensionCount,
    _In_reads_(dimensionCount) const UINT* sizes,
    _In_reads_opt_(dimensionCount) const UINT* strides
    )
{
    UINT elementSizeInBytes = 0;
    switch (dataType)
    {
    case DML_TENSOR_DATA_TYPE_FLOAT32:
    case DML_TENSOR_DATA_TYPE_UINT32:
    case DML_TENSOR_DATA_TYPE_INT32:
        elementSizeInBytes = 4;
        break;

    case DML_TENSOR_DATA_TYPE_FLOAT16:
    case DML_TENSOR_DATA_TYPE_UINT16:
    case DML_TENSOR_DATA_TYPE_INT16:
        elementSizeInBytes = 2;
        break;

    case DML_TENSOR_DATA_TYPE_UINT8:
    case DML_TENSOR_DATA_TYPE_INT8:
        elementSizeInBytes = 1;
        break;

    default:
        return 0; // Invalid data type
    }

    UINT64 minimumImpliedSizeInBytes = 0;
    if (!strides)
    {
        minimumImpliedSizeInBytes = sizes[0];
        for (UINT i = 1; i < dimensionCount; ++i)
        {
            minimumImpliedSizeInBytes *= sizes[i];
        }
        minimumImpliedSizeInBytes *= elementSizeInBytes;
    }
    else
    {
        UINT indexOfLastElement = 0;
        for (UINT i = 0; i < dimensionCount; ++i)
        {
            indexOfLastElement += (sizes[i] - 1) * strides[i];
        }

        minimumImpliedSizeInBytes = (indexOfLastElement + 1) * elementSizeInBytes;
    }

    // Round up to the nearest 4 bytes.
    minimumImpliedSizeInBytes = (minimumImpliedSizeInBytes + 3) & ~3ui64;

    return minimumImpliedSizeInBytes;
}

int __cdecl wmain(int /*argc*/, char ** /*argv*/)
{
    com_ptr<ID3D12Device> d3D12Device;
    com_ptr<ID3D12CommandQueue> commandQueue;
    com_ptr<ID3D12CommandAllocator> commandAllocator;
    com_ptr<ID3D12GraphicsCommandList> commandList;
    bool float16Convolution = true;

    // Set up Direct3D 12.
    InitializeDirect3D12(d3D12Device, commandQueue, commandAllocator, commandList);

    // Create the DirectML device.

    DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;

#if defined (_DEBUG)
    // If the project is in a debug build, then enable debugging via DirectML debug layers with this flag.
    dmlCreateDeviceFlags |= DML_CREATE_DEVICE_FLAG_DEBUG;
#endif

    com_ptr<IDMLDevice> dmlDevice;
    check_hresult(DMLCreateDevice(
        d3D12Device.get(),
        dmlCreateDeviceFlags,
        __uuidof(dmlDevice),
        dmlDevice.put_void()));

    const int dimensions = 2;

    convolution_shape shape;

    shape.m_batchCount = 1;
    shape.m_featureCount = 1;

    for (int i = 0; i < dimensions; i++) {
        shape.m_inputSize[i] = 5;
        shape.m_kernelSize[i] = 3;
        shape.m_startPadding[i] = 0;
        shape.m_endPadding[i] = 0;
        shape.m_kernelStride[i] = 1;
    }

    const int targetProductSums = 64;

    shape.m_channelCount = (int) (targetProductSums / (shape.m_kernelSize[0] * shape.m_kernelSize[1]));

    int productSums = shape.m_channelCount * (shape.m_kernelSize[0] * shape.m_kernelSize[1]);

    convolution_parameters<float> cp(shape);

    UINT batchCount = 1;
    UINT channels = shape.m_channelCount;
    UINT inputWidth = shape.m_inputSize[1];
    UINT inputHeight = shape.m_inputSize[0];
    UINT kernelWidth = shape.m_kernelSize[1];
    UINT kernelHeight = shape.m_kernelSize[0];
    UINT features = shape.m_featureCount;
    UINT kernelStride[dimensions] = { 1, 1 };
    UINT startPadding[dimensions] = { shape.m_startPadding[0], shape.m_startPadding[1] };
    UINT endPadding[dimensions] = { shape.m_endPadding[0], shape.m_endPadding[1] };
    UINT outputPadding[dimensions] = { 0, 0 };
    UINT dilations[dimensions] = { 0, 0 };

    UINT filterTensorSizes[4] = { features, channels, kernelWidth, kernelHeight };
    UINT filterTensorElementCount = filterTensorSizes[0] * filterTensorSizes[1] * filterTensorSizes[2] * filterTensorSizes[3];

    UINT outputTensorSizes[4] = { batchCount, features, cp.m_constants.m_outputSize[2], cp.m_constants.m_outputSize[3]};
    UINT outputTensorElementCount = outputTensorSizes[0] * outputTensorSizes[1] * outputTensorSizes[2] * outputTensorSizes[3];

    UINT kernelSizes[dimensions] = { kernelWidth, kernelHeight };

    DML_BUFFER_TENSOR_DESC dmlInputBufferTensorDesc = {};
    dmlInputBufferTensorDesc.DataType = (float16Convolution ? DML_TENSOR_DATA_TYPE_FLOAT16 : DML_TENSOR_DATA_TYPE_FLOAT32);
    dmlInputBufferTensorDesc.Flags = DML_TENSOR_FLAG_NONE;
    dmlInputBufferTensorDesc.DimensionCount = ARRAYSIZE(cp.m_constants.m_inputSize);
    dmlInputBufferTensorDesc.Sizes = cp.m_constants.m_inputSize;
    dmlInputBufferTensorDesc.Strides = nullptr;
    dmlInputBufferTensorDesc.TotalTensorSizeInBytes = (cp.m_constants.m_inputElementCount * (float16Convolution ? 2 : 4));
    // goofy DML rounds up to DWORD
    dmlInputBufferTensorDesc.TotalTensorSizeInBytes = ((dmlInputBufferTensorDesc.TotalTensorSizeInBytes + 3)) & ~3UL;

    DML_TENSOR_DESC dmlInputTensorDesc{};
    dmlInputTensorDesc.Type = DML_TENSOR_TYPE_BUFFER;
    dmlInputTensorDesc.Desc = &dmlInputBufferTensorDesc;

    DML_BUFFER_TENSOR_DESC dmlFilterBufferTensorDesc = {};
    dmlFilterBufferTensorDesc.DataType = (float16Convolution ? DML_TENSOR_DATA_TYPE_FLOAT16 : DML_TENSOR_DATA_TYPE_FLOAT32);;
    dmlFilterBufferTensorDesc.Flags = DML_TENSOR_FLAG_NONE;
    dmlFilterBufferTensorDesc.DimensionCount = ARRAYSIZE(cp.m_constants.m_filterSize);
    dmlFilterBufferTensorDesc.Sizes = cp.m_constants.m_filterSize;
    dmlFilterBufferTensorDesc.Strides = nullptr;
    dmlFilterBufferTensorDesc.TotalTensorSizeInBytes = (cp.m_constants.m_filterElementCount * (float16Convolution ? 2 : 4));
    // goofy DML rounds up to DWORD
    dmlFilterBufferTensorDesc.TotalTensorSizeInBytes = ((dmlFilterBufferTensorDesc.TotalTensorSizeInBytes + 3)) & ~3UL;

    DML_TENSOR_DESC dmlFilterTensorDesc{};
    dmlFilterTensorDesc.Type = DML_TENSOR_TYPE_BUFFER;
    dmlFilterTensorDesc.Desc = &dmlFilterBufferTensorDesc;

    DML_BUFFER_TENSOR_DESC dmlOutputBufferTensorDesc = {};
    dmlOutputBufferTensorDesc.DataType = (float16Convolution ? DML_TENSOR_DATA_TYPE_FLOAT16 : DML_TENSOR_DATA_TYPE_FLOAT32);;
    dmlOutputBufferTensorDesc.Flags = DML_TENSOR_FLAG_NONE;
    dmlOutputBufferTensorDesc.DimensionCount = ARRAYSIZE(cp.m_constants.m_outputSize);
    dmlOutputBufferTensorDesc.Sizes = cp.m_constants.m_outputSize;
    dmlOutputBufferTensorDesc.Strides = nullptr;
    dmlOutputBufferTensorDesc.TotalTensorSizeInBytes = (cp.m_constants.m_outputElementCount * (float16Convolution ? 2 : 4));
    // goofy DML rounds up to DWORD
    dmlOutputBufferTensorDesc.TotalTensorSizeInBytes = ((dmlFilterBufferTensorDesc.TotalTensorSizeInBytes + 3)) & ~3UL;

    DML_TENSOR_DESC dmlOutputTensorDesc{};
    dmlOutputTensorDesc.Type = DML_TENSOR_TYPE_BUFFER;
    dmlOutputTensorDesc.Desc = &dmlOutputBufferTensorDesc;

    com_ptr<IDMLOperator> dmlOperator;

    {
        DML_CONVOLUTION_OPERATOR_DESC dmlConvolutionOperatorDesc{};
        dmlConvolutionOperatorDesc.InputTensor = &dmlInputTensorDesc;
        dmlConvolutionOperatorDesc.FilterTensor = &dmlFilterTensorDesc;
        dmlConvolutionOperatorDesc.BiasTensor = nullptr;
        dmlConvolutionOperatorDesc.OutputTensor = &dmlOutputTensorDesc;
        dmlConvolutionOperatorDesc.Mode = DML_CONVOLUTION_MODE_CONVOLUTION;
        dmlConvolutionOperatorDesc.Direction = DML_CONVOLUTION_DIRECTION_FORWARD;
        dmlConvolutionOperatorDesc.DimensionCount = 2;
        dmlConvolutionOperatorDesc.Strides = kernelStride;
        dmlConvolutionOperatorDesc.Dilations = dilations;
        dmlConvolutionOperatorDesc.StartPadding = startPadding;
        dmlConvolutionOperatorDesc.EndPadding = endPadding;
        dmlConvolutionOperatorDesc.OutputPadding = outputPadding;
        dmlConvolutionOperatorDesc.GroupCount = batchCount;
        dmlConvolutionOperatorDesc.FusedActivation = NULL;


        // Like Direct3D 12, these DESC structs don't need to be long-lived. This means, for example, that it's safe to place
        // the DML_OPERATOR_DESC (and all the subobjects it points to) on the stack, since they're no longer needed after
        // CreateOperator returns.
        DML_OPERATOR_DESC dmlOperatorDesc{};
        dmlOperatorDesc.Type = DML_OPERATOR_CONVOLUTION;
        dmlOperatorDesc.Desc = &dmlConvolutionOperatorDesc;

         check_hresult(dmlDevice->CreateOperator(
            &dmlOperatorDesc,
            __uuidof(dmlOperator),
            dmlOperator.put_void()));
    }

    // Compile the operator into an object that can be dispatched to the GPU. In this step, DirectML performs operator
    // fusion and just-in-time (JIT) compilation of shader bytecode, then compiles it into a Direct3D 12 pipeline state object (PSO).
    // The resulting compiled operator is a baked, optimized form of an operator suitable for execution on the GPU.

    com_ptr<IDMLCompiledOperator> dmlCompiledOperator;
    check_hresult(dmlDevice->CompileOperator(
        dmlOperator.get(),
        DML_EXECUTION_FLAG_NONE,
        __uuidof(dmlCompiledOperator),
        dmlCompiledOperator.put_void()));

    com_ptr<IDMLOperatorInitializer> dmlOperatorInitializer;
    IDMLCompiledOperator* dmlCompiledOperators[] = { dmlCompiledOperator.get() };
    check_hresult(dmlDevice->CreateOperatorInitializer(
        ARRAYSIZE(dmlCompiledOperators),
        dmlCompiledOperators,
        __uuidof(dmlOperatorInitializer),
        dmlOperatorInitializer.put_void()));

    // Query the operator for the required size (in descriptors) of its binding table.
    // You need to initialize an operator exactly once before it can be executed, and
    // the two stages require different numbers of descriptors for binding. For simplicity,
    // we create a single descriptor heap that's large enough to satisfy them both.
    DML_BINDING_PROPERTIES initializeBindingProperties = dmlOperatorInitializer->GetBindingProperties();
    DML_BINDING_PROPERTIES executeBindingProperties = dmlCompiledOperator->GetBindingProperties();
    UINT descriptorCount = std::max(
        initializeBindingProperties.RequiredDescriptorCount,
        executeBindingProperties.RequiredDescriptorCount);

    // Create descriptor heaps.
    com_ptr<ID3D12DescriptorHeap> descriptorHeap;

    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = descriptorCount;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    check_hresult(d3D12Device->CreateDescriptorHeap(
        &descriptorHeapDesc,
        _uuidof(descriptorHeap),
        descriptorHeap.put_void()));

    // Set the descriptor heap(s).
    ID3D12DescriptorHeap* d3D12DescriptorHeaps[] = { descriptorHeap.get() };
    commandList->SetDescriptorHeaps(ARRAYSIZE(d3D12DescriptorHeaps), d3D12DescriptorHeaps);

    // Create a binding table over the descriptor heap we just created.
    DML_BINDING_TABLE_DESC dmlBindingTableDesc{};
    dmlBindingTableDesc.Dispatchable = dmlOperatorInitializer.get();
    dmlBindingTableDesc.CPUDescriptorHandle = descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    dmlBindingTableDesc.GPUDescriptorHandle = descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    dmlBindingTableDesc.SizeInDescriptors = descriptorCount;

    com_ptr<IDMLBindingTable> dmlBindingTable;
    check_hresult(dmlDevice->CreateBindingTable(
        &dmlBindingTableDesc,
        __uuidof(dmlBindingTable),
        dmlBindingTable.put_void()));

    // Create the temporary and persistent resources that are necessary for executing an operator.

    // The temporary resource is scratch memory (used internally by DirectML), whose contents you don't need to define.
    // The persistent resource is long-lived, and you need to initialize it using the IDMLOperatorInitializer.

    UINT64 temporaryResourceSize = std::max(
        initializeBindingProperties.TemporaryResourceSize,
        executeBindingProperties.TemporaryResourceSize);
    UINT64 persistentResourceSize = executeBindingProperties.PersistentResourceSize;

    // Bind and initialize the operator on the GPU.

    com_ptr<ID3D12Resource> temporaryBuffer;
    if (temporaryResourceSize != 0)
    {
        check_hresult(d3D12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(temporaryResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            __uuidof(temporaryBuffer),
            temporaryBuffer.put_void()));

        DML_BUFFER_BINDING bufferBinding{ temporaryBuffer.get(), 0, temporaryResourceSize };
        DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
        dmlBindingTable->BindTemporaryResource(&bindingDesc);
    }

    com_ptr<ID3D12Resource> persistentBuffer;
    if (persistentResourceSize != 0)
    {
        check_hresult(d3D12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(persistentResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            __uuidof(persistentBuffer),
            persistentBuffer.put_void()));

        // The persistent resource should be bound as the output to the IDMLOperatorInitializer.
        DML_BUFFER_BINDING bufferBinding{ persistentBuffer.get(), 0, persistentResourceSize };
        DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
        dmlBindingTable->BindOutputs(1, &bindingDesc);
    }

    // The command recorder is a stateless object that records Dispatches into an existing Direct3D 12 command list.
    com_ptr<IDMLCommandRecorder> dmlCommandRecorder;
    check_hresult(dmlDevice->CreateCommandRecorder(
        __uuidof(dmlCommandRecorder),
        dmlCommandRecorder.put_void()));

    // Record execution of the operator initializer.
    dmlCommandRecorder->RecordDispatch(
        commandList.get(),
        dmlOperatorInitializer.get(),
        dmlBindingTable.get());

    // Close the Direct3D 12 command list, and submit it for execution as you would any other command list. You could
    // in principle record the execution into the same command list as the initialization, but you need only to Initialize
    // once, and typically you want to Execute an operator more frequently than that.
    CloseExecuteResetWait(d3D12Device, commandQueue, commandAllocator, commandList);

    // 
    // Bind and execute the operator on the GPU.
    // 

    commandList->SetDescriptorHeaps(ARRAYSIZE(d3D12DescriptorHeaps), d3D12DescriptorHeaps);

    // Reset the binding table to bind for the operator we want to execute (it was previously used to bind for the
    // initializer).

    dmlBindingTableDesc.Dispatchable = dmlCompiledOperator.get();

    check_hresult(dmlBindingTable->Reset(&dmlBindingTableDesc));

    if (temporaryResourceSize != 0)
    {
        DML_BUFFER_BINDING bufferBinding{ temporaryBuffer.get(), 0, temporaryResourceSize };
        DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
        dmlBindingTable->BindTemporaryResource(&bindingDesc);
    }

    if (persistentResourceSize != 0)
    {
        DML_BUFFER_BINDING bufferBinding{ persistentBuffer.get(), 0, persistentResourceSize };
        DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
        dmlBindingTable->BindPersistentResource(&bindingDesc);
    }

    // Create tensor buffers for upload/input/output/readback of the tensor elements.

    UINT64 inputTensorBufferSize{ dmlInputBufferTensorDesc.TotalTensorSizeInBytes };
    UINT64 filterTensorBufferSize{ dmlFilterBufferTensorDesc.TotalTensorSizeInBytes };
    UINT64 outputTensorBufferSize{ dmlOutputBufferTensorDesc.TotalTensorSizeInBytes };

    com_ptr<ID3D12Resource> uploadInputBuffer;
    check_hresult(d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(inputTensorBufferSize),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        __uuidof(uploadInputBuffer),
        uploadInputBuffer.put_void()));

    com_ptr<ID3D12Resource> uploadFilterBuffer;
    check_hresult(d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(filterTensorBufferSize),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        __uuidof(uploadFilterBuffer),
        uploadFilterBuffer.put_void()));

    com_ptr<ID3D12Resource> inputBuffer;
    check_hresult(d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(inputTensorBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        __uuidof(inputBuffer),
        inputBuffer.put_void()));

    com_ptr<ID3D12Resource> filterBuffer;
    check_hresult(d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(filterTensorBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        __uuidof(filterBuffer),
        filterBuffer.put_void()));

    std::wcout << std::fixed; std::wcout.precision(2);

    const static int floatPrecision = 23;
    const static int halfPrecision = 10;
    double targetProduct = 1.5;
    double estimatedSum = targetProduct * productSums;
    const static int bitsUsedForAccumulation = (int) (log2(estimatedSum) + 0.99);
    double estimatedExponent = log2(estimatedSum);
    double floatSumUlp = pow(2.0f, estimatedExponent - floatPrecision );
    double halfSumUlp = pow(2.0f, estimatedExponent - halfPrecision );
    double estimatedFloatMaxError = floatSumUlp * sqrt(productSums) / 2;
    double estimatedHalfMaxError = halfSumUlp * sqrt(productSums) / 2;

    printf("estimated float max error = %1.8lf\n", estimatedFloatMaxError);
    printf("estimated half max error = %1.8lf\n", estimatedHalfMaxError);

    assert(shape.m_batchCount == 1 );

    srand(0);   // use same seed to generate same data over different runs

    // We want the product to be targetProduct +/- (ulp/2)
    // NOTE: the delta will turn into (2*delta) in product thus ... floatSumUlp / 4
    double delta = floatSumUlp / 4;
    double squareRootTarget = sqrt(targetProduct);
    for (int i = 0; i < cp.m_buffers.m_input.size(); i++)
        cp.m_buffers.m_input[i] = (float)(squareRootTarget + (randd() * delta));
    for (int i = 0; i < cp.m_buffers.m_filter.size(); i++)
        cp.m_buffers.m_filter[i] = (float)(squareRootTarget  - (randd() * delta));

    {
#if 0
        std::wcout << L"input tensor: ";
#endif
#if 0
        for (auto & element : inputTensorElementArray)
        {
            element = 1.618f;
#if 0
            std::wcout << element << L' ';
#endif
        };
#endif
#if 0
        std::wcout << std::endl;
#endif

        std::vector< uint16_t> inputData16;
        std::vector< uint16_t> filterData16;
        
        const void * inputData = cp.m_buffers.m_input.data();
        const void * filterData = cp.m_buffers.m_filter.data();

        if (float16Convolution) {
            inputData16.resize(cp.m_buffers.m_input.size());
            filterData16.resize(cp.m_buffers.m_filter.size());

            for(int i = 0; i < cp.m_buffers.m_input.size(); i++)
                inputData16[i] = Float16Compressor::compress(cp.m_buffers.m_input[i]);
            inputData = inputData16.data();

            for (int i = 0; i < cp.m_buffers.m_filter.size(); i++)
                filterData16[i] = Float16Compressor::compress(cp.m_buffers.m_filter[i]);
            filterData = filterData16.data();
        }

        D3D12_SUBRESOURCE_DATA tensorSubresourceData{};
        tensorSubresourceData.pData = inputData;
        tensorSubresourceData.RowPitch = inputTensorBufferSize;
        tensorSubresourceData.SlicePitch = tensorSubresourceData.RowPitch;

        // Upload the input tensor to the GPU.
        ::UpdateSubresources(
            commandList.get(),
            inputBuffer.get(),
            uploadInputBuffer.get(),
            0,
            0,
            1,
            &tensorSubresourceData);

        commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                inputBuffer.get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                )
            );

        // Upload the filter tensor to the GPU
        tensorSubresourceData.pData = filterData;
        tensorSubresourceData.RowPitch = filterTensorBufferSize;
        tensorSubresourceData.SlicePitch = tensorSubresourceData.RowPitch;

        ::UpdateSubresources(
            commandList.get(),
            filterBuffer.get(),
            uploadFilterBuffer.get(),
            0,
            0,
            1,
            &tensorSubresourceData);

        commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                filterBuffer.get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            )
        );
    }

    DML_BUFFER_BINDING inputBufferBinding{ inputBuffer.get(), 0, inputTensorBufferSize };
    DML_BUFFER_BINDING filterBufferBinding{ filterBuffer.get(), 0, filterTensorBufferSize };
    DML_BINDING_DESC descs[3];
    descs[0] = { DML_BINDING_TYPE_BUFFER, &inputBufferBinding };
    descs[1] = { DML_BINDING_TYPE_BUFFER, &filterBufferBinding };
    descs[2] = { DML_BINDING_TYPE_NONE, nullptr };

    dmlBindingTable->BindInputs(3, descs);

    com_ptr<ID3D12Resource> outputBuffer;
    check_hresult(d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(outputTensorBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        __uuidof(outputBuffer),
        outputBuffer.put_void()));

    DML_BUFFER_BINDING outputBufferBinding{ outputBuffer.get(), 0, outputTensorBufferSize };
    DML_BINDING_DESC outputBindingDesc{ DML_BINDING_TYPE_BUFFER, &outputBufferBinding };
    dmlBindingTable->BindOutputs(1, &outputBindingDesc);

    const int dispatchCount = 1;
    // Record execution of the compiled operator.
    for (int i = 0; i < dispatchCount; i++) {
        dmlCommandRecorder->RecordDispatch(commandList.get(), dmlCompiledOperator.get(), dmlBindingTable.get());

        commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::UAV(
                inputBuffer.get()
            )
        );

        commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::UAV(
                filterBuffer.get()
            )
        );

        commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::UAV(
                outputBuffer.get()
            )
        );

    }

    double seconds = CloseExecuteResetWait(d3D12Device, commandQueue, commandAllocator, commandList);

    // The output buffer now contains the result of the identity operator,
    // so read it back if you want the CPU to access it.

    com_ptr<ID3D12Resource> readbackBuffer;
    check_hresult(d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(outputTensorBufferSize),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        __uuidof(readbackBuffer),
        readbackBuffer.put_void()));

    commandList->ResourceBarrier(
        1,
        &CD3DX12_RESOURCE_BARRIER::Transition(
            outputBuffer.get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE
            )
        );

    commandList->CopyResource(readbackBuffer.get(), outputBuffer.get());

    CloseExecuteResetWait(d3D12Device, commandQueue, commandAllocator, commandList);

    D3D12_RANGE tensorBufferRange{ 0, outputTensorBufferSize };

    void* outputData;
    check_hresult(readbackBuffer->Map(0, &tensorBufferRange, &outputData));

    convolution_evaluate(cp);

    static bool show_output = true;

     bool valuesMatch = true;
    float maxError = 0.0f;
    float maxPercentageError = 0.0f;

    float * output32 = (float *)outputData;
    uint16_t * output16 = (uint16_t *)outputData;

    for (int i = 0; i < cp.m_buffers.m_output.size(); i++)
    {
        float calculated = (float16Convolution ? Float16Compressor::decompress(output16[i]) : output32[i]);
        float expected = cp.m_buffers.m_output[i];

        if (show_output)
            printf("[%d] %4.10f %4.10f\n", i, calculated, expected);

        float error = abs(calculated - expected);
        float percentageError = (error / abs(expected)) * 100.0f;

        maxError = (error > maxError ? error : maxError);
        maxPercentageError = (percentageError > maxPercentageError ? percentageError : maxPercentageError);
    }

    if (show_output)
        std::wcout << std::endl;

    D3D12_RANGE emptyRange{ 0, 0 };
    readbackBuffer->Unmap(0, &emptyRange);

    printf("max error = %4.10f\n", maxError);
    printf("max percentage error = %2.6f\n", maxPercentageError);

#if 0
    double dataSize = (double) outputTensorElementCount * (double) sizeof(FLOAT);
    double ioSize = dataSize * 3.0 * (double) dispatchCount;
    double throughput = ioSize / seconds;
    double flops = (double)outputTensorElementCount * (double) dispatchCount / seconds;

    printf("time = %fs\n", seconds);
    printf("throughput = %f GB/sec\n", throughput / (1024.0 * 1024.0 * 1024.0));
    printf("%f GFlops\n", flops / (1024.0 * 1024.0 * 1024.0));
#endif

    printf("eval time = %fms\n", (seconds / (double)dispatchCount) * 1000.0);
}
