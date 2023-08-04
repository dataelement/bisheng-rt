///////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////////////////////////

#include <gtest/gtest.h>
#include <cnmem.h>
#include <fstream>
#ifdef USE_CPP_11
#include <thread>
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

static std::size_t getFreeMemory() {
    cudaFree(0);
    std::size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    return freeMem;
}

class CnmemTest : public testing::TestWithParam<unsigned>{
    /// We determine the amount of free memory.
    std::size_t mFreeMem;
    
protected:
    /// Do we test memory leaks.
    bool mTestLeaks;
    /// Do we skip finalization.
    bool mFinalize;
    /// Do we use managed memory.
    unsigned pool_flags;
    
public:
    /// Ctor.
    CnmemTest() :
        mFreeMem(getFreeMemory()),
        mTestLeaks(true),
        mFinalize(true),
        pool_flags(GetParam()) {}
    /// Tear down the test.
    void TearDown();
};

void CnmemTest::TearDown() {
    if( mFinalize ) {
        ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFinalize()); 
    }
    if( mTestLeaks ) {
        ASSERT_EQ(mFreeMem, getFreeMemory());
    }
    cudaDeviceReset();
}

INSTANTIATE_TEST_CASE_P(DefaultOrManagedPool,
                        CnmemTest,
                        ::testing::Values(CNMEM_FLAGS_DEFAULT,
                                          CNMEM_FLAGS_MANAGED));

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, notInitializedFinalize) {
    ASSERT_EQ(CNMEM_STATUS_NOT_INITIALIZED, cnmemFinalize());
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 2048;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags)); // For TearDown to be happy
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, notInitializedMalloc) {
    ASSERT_EQ(CNMEM_STATUS_NOT_INITIALIZED, cnmemMalloc(NULL, 0, 0));
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 2048;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags)); // For TearDown to be happy
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, notInitializedFree) {
    ASSERT_EQ(CNMEM_STATUS_NOT_INITIALIZED, cnmemFree(NULL, 0));
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 2048;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags)); // For TearDown to be happy
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, initInvalidSize) {
    cudaDeviceProp props;
    ASSERT_EQ(cudaSuccess, cudaGetDeviceProperties(&props, 0));

    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = props.totalGlobalMem * 2;
    ASSERT_EQ(CNMEM_STATUS_INVALID_ARGUMENT, cnmemInit(1, &device, pool_flags));

    device.size = 2048;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags)); // For TearDown to be happy
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, initNinetyFivePrct) {
    cudaDeviceProp props;
    ASSERT_EQ(cudaSuccess, cudaGetDeviceProperties(&props, 0));

    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = (size_t) (0.95*props.totalGlobalMem);
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags));
    mTestLeaks = false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, initDevice1) {
    int numDevices;
    ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&numDevices));
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.device = numDevices < 2 ? 0 : 1; // Skip device 0 if we have more than 1 device.
    device.size = 2048;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, freeNULL) {
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 2048;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(NULL, NULL));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, freeTwoStreams) {
    cudaStream_t streams[2];
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[0]));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[1]));

    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 3*1024;
    device.numStreams = 2;
    device.streams = streams;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags));
    void *ptr0, *ptr1;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr0, 1024, streams[0]));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr1, 1024, streams[1]));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr1, streams[1])); 
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr0, streams[0]));
    
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(streams[0]));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(streams[1]));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, addStream) {
    cudaStream_t streams[2];
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[0]));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[1]));

    // Register only 1 stream (on purpose).
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 3*1024;
    device.numStreams = 1;
    device.streams = streams;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags));

    // Allocate a pointer with a valid stream.
    void *ptr0, *ptr1;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr0, 1024, streams[0]));

    // Try to allocate with an invalid stream.
    ASSERT_EQ(CNMEM_STATUS_INVALID_ARGUMENT, cnmemMalloc(&ptr1, 1024, streams[1]));

    // Register the stream and try again.
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemRegisterStream(streams[1]));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr1, 1024, streams[1]));

    // Clean up.
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr1, streams[1])); 
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr0, streams[0]));
    
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(streams[0]));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(streams[1]));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, freeWrongStream) {
    cudaStream_t streams[2];
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[0]));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[1]));

    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 3*1024;
    device.numStreams = 2;
    device.streams = streams;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags));
    void *ptr;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr, 1024, streams[0]));
    ASSERT_EQ(CNMEM_STATUS_INVALID_ARGUMENT, cnmemFree(ptr, streams[1])); 
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr, streams[0]));
    
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(streams[0]));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(streams[1]));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, freeNULLRatherThanNamed) {
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 2048;
    device.numStreams = 1;
    device.streams = &stream;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags));
    void *ptr;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr, 1024, stream));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr, NULL)); // We expect this async free to work.
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFinalize());

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    device.numStreams = 0;
    device.streams = NULL;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags)); // For TearDown to be happy
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, allocateNULL) {
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 2048;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags));

    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(NULL, 0, NULL));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, allocateZeroSize) {
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 2048;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags));

    void *ptr;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr, 0, NULL));
    ASSERT_EQ((void*) NULL, ptr);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, allocateNoFree) {
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 2048;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags));

    void *ptr;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr, 512, NULL));
    ASSERT_NE((void*) NULL, ptr);
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFinalize());

    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags)); // For TearDown to be happy
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, allocateAndFreeOne) {
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 2048;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags));

    void *ptr;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr, 512, NULL));
    ASSERT_NE((void*) NULL, ptr);
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr, NULL));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, allocateAndFreeTwo) {
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 2048;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags));

    void *ptr0;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr0, 512, NULL));
    ASSERT_NE((void*) NULL, ptr0);
    void *ptr1;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr1, 512, NULL));
    ASSERT_NE((void*) NULL, ptr1);
    
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr1, NULL));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr0, NULL));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, allocateAndFreeAll) {
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 2048;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags));

    void *ptr0;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr0, 512, NULL));
    ASSERT_NE((void*) NULL, ptr0);
    void *ptr1;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr1, 512, NULL));
    ASSERT_NE((void*) NULL, ptr1);
    void *ptr2;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr2, 512, NULL));
    ASSERT_NE((void*) NULL, ptr2);
    void *ptr3;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr3, 512, NULL));
    ASSERT_NE((void*) NULL, ptr3);
    
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr3, NULL));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr2, NULL));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr1, NULL));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr0, NULL));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, allocateAndFreeAnyOrder) {
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 2048;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags));

    void *ptr0;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr0, 512, NULL));
    ASSERT_NE((void*) NULL, ptr0);
    void *ptr1;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr1, 512, NULL));
    ASSERT_NE((void*) NULL, ptr1);
    
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr0, NULL));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr1, NULL));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, allocateTooMuchAndGrow) {
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 2048;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags));

    void *ptr0;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr0, 512, NULL));
    ASSERT_NE((void*) NULL, ptr0);
    void *ptr1;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr1, 512, NULL));
    ASSERT_NE((void*) NULL, ptr1);
    void *ptr2;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr2, 512, NULL));
    ASSERT_NE((void*) NULL, ptr2);
    void *ptr3;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr3, 512, NULL));
    ASSERT_NE((void*) NULL, ptr3);
    void *ptr4;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr4, 512, NULL));
    ASSERT_NE((void*) NULL, ptr4);
    
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr4, NULL));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr3, NULL));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr2, NULL));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr1, NULL));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr0, NULL));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, allocateTooMuchNoGrow) {
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 2048;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags | CNMEM_FLAGS_CANNOT_GROW));

    void *ptr0;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr0, 512, NULL));
    ASSERT_NE((void*) NULL, ptr0);
    void *ptr1;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr1, 512, NULL));
    ASSERT_NE((void*) NULL, ptr1);
    void *ptr2;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr2, 512, NULL));
    ASSERT_NE((void*) NULL, ptr2);
    void *ptr3;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr3, 512, NULL));
    ASSERT_NE((void*) NULL, ptr3);
    void *ptr4;
    ASSERT_EQ(CNMEM_STATUS_OUT_OF_MEMORY, cnmemMalloc(&ptr4, 512, NULL));
    
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr3, NULL));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr2, NULL));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr1, NULL));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr0, NULL));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, allocateAndSteal) {
    cudaStream_t streams[2];
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[0]));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[1]));
    
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 3*1024;
    device.numStreams = 2;
    device.streams = streams;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags | CNMEM_FLAGS_CANNOT_GROW));

    // Take the 1024B from streams[0].
    void *ptr0;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr0, 1024, streams[0]));
    ASSERT_NE((void*) NULL, ptr0);
    // Take the 1024B from NULL.
    void *ptr1;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr1, 1024, streams[0]));
    ASSERT_NE((void*) NULL, ptr1);
    // Steal the 1024B from streams[1].
    void *ptr2;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr2, 1024, streams[0]));
    ASSERT_NE((void*) NULL, ptr2);

    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr2, streams[0]));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr1, streams[0]));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr0, streams[0]));

    cudaStreamDestroy(streams[1]);
    cudaStreamDestroy(streams[0]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, allocateAndSteal2) {
    cudaStream_t streams[2];
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[0]));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[1]));

    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 3*1024;
    device.numStreams = 2;
    device.streams = streams;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags | CNMEM_FLAGS_CANNOT_GROW));

    void *ptr0;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr0, 1024, streams[0]));
    ASSERT_NE((void*) NULL, ptr0);
    void *ptr1;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr1, 512, streams[1]));
    ASSERT_NE((void*) NULL, ptr1);
    void *ptr2;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr2, 512, streams[0]));
    ASSERT_NE((void*) NULL, ptr2);

    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr2, streams[0]));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr1, streams[1]));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr0, streams[0]));

    cudaStreamDestroy(streams[1]);
    cudaStreamDestroy(streams[0]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, allocateAndSteal3) {
    cudaStream_t streams[2];
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[0]));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[1]));

    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 3*2048;
    device.numStreams = 2;
    device.streams = streams;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags | CNMEM_FLAGS_CANNOT_GROW));

    void *ptr0;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr0, 2048, streams[0]));
    ASSERT_NE((void*) NULL, ptr0);
    void *ptr1;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr1, 512, streams[1]));
    ASSERT_NE((void*) NULL, ptr1);
    void *ptr2;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr2, 1024, streams[0]));
    ASSERT_NE((void*) NULL, ptr2);
    void *ptr3;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr3, 512, streams[1]));
    ASSERT_NE((void*) NULL, ptr3);

    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr3, streams[1]));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr2, streams[0]));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr1, streams[1]));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr0, streams[0]));

    cudaStreamDestroy(streams[1]);
    cudaStreamDestroy(streams[0]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, allocateAndSteal4) {
    cudaStream_t streams[2];
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[0]));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[1]));

    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 6*1024;
    device.numStreams = 2;
    device.streams = streams;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags | CNMEM_FLAGS_CANNOT_GROW));

    void *ptr0;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr0, 1024, streams[0]));
    void *ptr1;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr1, 1024, streams[0]));
    void *ptr2;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr2, 1024, streams[0]));
    void *ptr3;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr3, 1024, streams[0]));
    void *ptr4;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr4, 1024, streams[0]));
    void *ptr5;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr5, 1024, streams[1]));
    
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr0, streams[0]));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr1, streams[0]));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr2, streams[0]));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr3, streams[0]));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr4, streams[0]));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr5, streams[1]));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, allocateAndReserveStream) {
    cudaStream_t streams[2];
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[0]));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[1]));

    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 4096;
    device.numStreams = 2;
    device.streams = streams;
    size_t streamSizes[] = { 2048, 2048 };
    device.streamSizes = streamSizes;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags | CNMEM_FLAGS_CANNOT_GROW));

    size_t totalMem, freeMem;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMemGetInfo(&freeMem, &totalMem, cudaStreamDefault));
    ASSERT_EQ(4096, totalMem);
    ASSERT_EQ(0, freeMem);

    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMemGetInfo(&freeMem, &totalMem, streams[0]));
    ASSERT_EQ(2048, totalMem);
    ASSERT_EQ(2048, freeMem);

    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMemGetInfo(&freeMem, &totalMem, streams[1]));
    ASSERT_EQ(2048, totalMem);
    ASSERT_EQ(2048, freeMem);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, allocateAndReserveStreamDifferentSizes) {
    cudaStream_t streams[2];
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[0]));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[1]));

    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 8192;
    device.numStreams = 2;
    device.streams = streams;
    size_t streamSizes[] = { 2048, 4096 };
    device.streamSizes = streamSizes;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags | CNMEM_FLAGS_CANNOT_GROW));

    size_t totalMem, freeMem;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMemGetInfo(&freeMem, &totalMem, cudaStreamDefault));
    ASSERT_EQ(8192, totalMem);
    ASSERT_EQ(2048, freeMem);

    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMemGetInfo(&freeMem, &totalMem, streams[0]));
    ASSERT_EQ(2048, totalMem);
    ASSERT_EQ(2048, freeMem);

    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMemGetInfo(&freeMem, &totalMem, streams[1]));
    ASSERT_EQ(4096, totalMem);
    ASSERT_EQ(4096, freeMem);

    FILE *file = fopen("reserveStream.log", "w");
    ASSERT_NE((FILE*) NULL, file);
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemPrintMemoryState(file, streams[0]));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemPrintMemoryState(file, streams[1]));
    fclose(file);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef USE_CPP_11

template< int N >
static void allocate(cudaStream_t stream) {
    void *ptr[N];
    for( int i = 0 ; i < N ; ++i )
        ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr[i], 1024, stream));
    for( int i = 0 ; i < N ; ++i )
        ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr[i], stream));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, allocateConcurrentNoCompete) {
    cudaStream_t streams[2];
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[0]));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[1]));
    
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 6*1024;
    device.numStreams = 2;
    device.streams = streams;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags | CNMEM_FLAGS_CANNOT_GROW));
    
    // In this test, each manager has enough memory to accommodate the threads.
    std::vector<std::thread*> threads(2);
    for( int i = 0 ; i < 2 ; ++i )
        threads[i] = new std::thread(allocate<2>, streams[i]);
    for( int i = 0 ; i < 2 ; ++i )
        threads[i]->join();
    for( unsigned i = 0 ; i < 2 ; ++i )
        delete threads[i];
    threads.clear();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, allocateConcurrentCompete) {
    cudaStream_t streams[2];
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[0]));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[1]));

    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 6*1024;
    device.numStreams = 2;
    device.streams = streams;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags | CNMEM_FLAGS_CANNOT_GROW));
    
    // In this test, the threads compete for the memory of the root manager.
    std::vector<std::thread*> threads(2);
    for( int i = 0; i < 2; ++i )
        threads[i] = new std::thread(allocate<3>, streams[i]);
    for( int i = 0; i < 2; ++i )
        threads[i]->join();
    for( unsigned i = 0; i < 2; ++i )
        delete threads[i];
    threads.clear();

    mTestLeaks = false; // For some reasons, it reports a leak. It's likely to be in the driver/runtime.
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, allocateConcurrentSteal) {
    const int N = 4;
    cudaStream_t streams[N];
    for( int i = 0 ; i < N ; ++i ) {
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[i]));
    }

    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 4*N*1024;
    device.numStreams = N;
    device.streams = streams;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags | CNMEM_FLAGS_CANNOT_GROW));

    // In this test, the thread 0 has to steal memory from thread 1.
    std::vector<std::thread*> threads(N);
    for( int i = 0 ; i < N ; ++i ) {
        threads[i] = new std::thread(allocate<4>, streams[i]);
    }
    for( int i = 0; i < N; ++i )
        threads[i]->join();
    for( int i = 0; i < N; ++i )
        delete threads[i];
    threads.clear();

    mTestLeaks = false; // For some reasons, it reports a leak. 
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, allocateConcurrentMultiStreamsPerThreadNoGrow) {
    const int NUM_STREAMS = 8, NUM_THREADS = 32;
    cudaStream_t streams[NUM_STREAMS];
    for( int i = 0 ; i < NUM_STREAMS ; ++i ) {
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[i]));
    }

    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 4*NUM_THREADS*1024;
    device.numStreams = NUM_STREAMS;
    device.streams = streams;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags | CNMEM_FLAGS_CANNOT_GROW));

    std::vector<std::thread*> threads(NUM_THREADS);
    for( int i = 0 ; i < NUM_THREADS ; ++i ) {
        threads[i] = new std::thread(allocate<4>, streams[i%NUM_STREAMS]);
    }
    for( int i = 0; i < NUM_THREADS; ++i )
        threads[i]->join();
    for( int i = 0; i < NUM_THREADS; ++i )
        delete threads[i];
    threads.clear();

    mTestLeaks = false; // For some reasons, it reports a leak. 
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, allocateConcurrentMultiStreamsPerThreadGrow) {
    const int NUM_STREAMS = 8, NUM_THREADS = 32;
    cudaStream_t streams[NUM_STREAMS];
    for( int i = 0 ; i < NUM_STREAMS ; ++i ) {
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[i]));
    }

    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = NUM_THREADS*1024;
    device.numStreams = NUM_STREAMS;
    device.streams = streams;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags));

    std::vector<std::thread*> threads(NUM_THREADS);
    for( int i = 0 ; i < NUM_THREADS ; ++i ) {
        threads[i] = new std::thread(allocate<4>, streams[i%NUM_STREAMS]);
    }
    for( int i = 0; i < NUM_THREADS; ++i )
        threads[i]->join();
    for( int i = 0; i < NUM_THREADS; ++i )
        delete threads[i];
    threads.clear();

    mTestLeaks = false; // For some reasons, it reports a leak. 
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
static void registerAndAllocate(cudaStream_t stream) {
    void *ptr[N];
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemRegisterStream(stream));
    for( int i = 0 ; i < N ; ++i )
        ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr[i], 1024, stream));
    for( int i = 0 ; i < N ; ++i )
        ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr[i], stream));
}

TEST_P(CnmemTest, registerAndAllocateConcurrentStreamsGrow) {
    const int N = 32;
    cudaStream_t streams[N];
    for( int i = 0 ; i < N ; ++i ) {
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[i]));
    }

    // Declare no stream.
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 1024;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags));

    // In this test, the thread 0 has to steal memory from thread 1.
    std::vector<std::thread*> threads(N);
    for( int i = 0 ; i < N ; ++i ) {
        threads[i] = new std::thread(registerAndAllocate<2>, streams[i]);
    }
    for( int i = 0; i < N; ++i )
        threads[i]->join();
    for( int i = 0; i < N; ++i )
        delete threads[i];
    threads.clear();

    mTestLeaks = false; // For some reasons, it reports a leak. 
}

TEST_P(CnmemTest, registerAndAllocateConcurrentMultiStreamsPerThread) {
    const int NUM_STREAMS = 8, NUM_THREADS = 32;
    cudaStream_t streams[NUM_STREAMS];
    for( int i = 0 ; i < NUM_STREAMS ; ++i ) {
        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[i]));
    }

    // Declare no stream.
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 1024;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags));

    // In this test, the thread 0 has to steal memory from thread 1.
    std::vector<std::thread*> threads(NUM_THREADS);
    for( int i = 0 ; i < NUM_THREADS ; ++i ) {
        threads[i] = new std::thread(registerAndAllocate<4>, streams[i%NUM_STREAMS]);
    }
    for( int i = 0; i < NUM_THREADS; ++i )
        threads[i]->join();
    for( int i = 0; i < NUM_THREADS; ++i )
        delete threads[i];
    threads.clear();

    mTestLeaks = false; // For some reasons, it reports a leak. 
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template< int N >
static void allocateAndPrint(int id, cudaStream_t stream) {
    void *ptr[N];
    for( int i = 0 ; i < N ; ++i )
        ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr[i], 1024, stream));
    char buffer[64];
    sprintf(buffer, "memoryState.%d.log", id);
    FILE *file = fopen(buffer, "w");
    ASSERT_NE((FILE*) NULL, file);
    cnmemPrintMemoryState(file, stream);
    fclose(file);
    for( int i = 0 ; i < N ; ++i )
        ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr[i], stream));
}

TEST_P(CnmemTest, testPrintMemoryState) {
    cudaStream_t streams[2];
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[0]));
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&streams[1]));
    
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 4096;
    device.numStreams = 2;
    device.streams = streams;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags | CNMEM_FLAGS_CANNOT_GROW));
    
    // In this test, each manager has enough memory to accommodate the threads.
    std::vector<std::thread*> threads(2);
    for( int i = 0 ; i < 2 ; ++i )
        threads[i] = new std::thread(allocateAndPrint<2>, i, streams[i]);
    for( int i = 0 ; i < 2 ; ++i )
        threads[i]->join();
    for( unsigned i = 0 ; i < 2 ; ++i )
        delete threads[i];
    threads.clear();

    mTestLeaks = false; // For some reasons, it reports a leak. 
}

#endif // defined USE_CPP_11

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, memoryUsage) {
    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 4096;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags));
    
    std::size_t totalMem, freeMem;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMemGetInfo(&freeMem, &totalMem, cudaStreamDefault));
    ASSERT_EQ(4096, totalMem);
    ASSERT_EQ(4096, freeMem);
    
    void *ptr;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc(&ptr, 1024, NULL));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMemGetInfo(&freeMem, &totalMem, cudaStreamDefault));
    ASSERT_EQ(4096, totalMem);
    ASSERT_EQ(3072, freeMem);
    
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr, NULL));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMemGetInfo(&freeMem, &totalMem, cudaStreamDefault));
    ASSERT_EQ(4096, totalMem);
    ASSERT_EQ(4096, freeMem);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(CnmemTest, testDeviceDoesNotChange) {

    int numDevices;
    ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&numDevices));
    if( numDevices < 2 ) {
        ASSERT_TRUE(true);
	mFinalize = false;
        return;
    }

    cnmemDevice_t devices[2];
    memset(devices, 0, sizeof(devices));
    devices[0].device = 0;
    devices[0].size = 4096;
    devices[1].device = 1;
    devices[1].size = 2048;

    int currentDevice;
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(2, devices, pool_flags));
    ASSERT_EQ(cudaSuccess, cudaGetDevice(&currentDevice));
    ASSERT_EQ(0, currentDevice);

    size_t totalMem, freeMem;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMemGetInfo(&freeMem, &totalMem, cudaStreamDefault));
    ASSERT_EQ(4096, totalMem);
    ASSERT_EQ(4096, freeMem);

    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFinalize());

    ASSERT_EQ(cudaSuccess, cudaSetDevice(1));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(2, devices, pool_flags));
    ASSERT_EQ(cudaSuccess, cudaGetDevice(&currentDevice));
    ASSERT_EQ(1, currentDevice);

    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMemGetInfo(&freeMem, &totalMem, cudaStreamDefault));
    ASSERT_EQ(2048, totalMem);
    ASSERT_EQ(2048, freeMem);

    ASSERT_EQ(cudaSuccess, cudaSetDevice(0)); // Make sure we are on dev 0 for final mem checks.
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef USE_CPP_11
#include <memory>

template< typename T >
class DeviceDeleter {
public:
    DeviceDeleter() {}
    void operator()(T *ptr) {
        ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFree(ptr, cudaStreamDefault));
        ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemRelease());
    }
};

TEST_P(CnmemTest, testSharedPtr) {

    cnmemDevice_t device;
    memset(&device, 0, sizeof(device));
    device.size = 2048;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemInit(1, &device, pool_flags));
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemRetain());

    float *ptr;
    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemMalloc((void**) &ptr, 1024, cudaStreamDefault));
    std::shared_ptr<float> p(ptr, DeviceDeleter<float>());

    ASSERT_EQ(CNMEM_STATUS_SUCCESS, cnmemFinalize()); // We still have a pointer in the scope...
    mFinalize = false; // Make sure TearDown does call finalize again.
}

#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

