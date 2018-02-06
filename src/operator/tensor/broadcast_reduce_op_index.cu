/*!
 *  Copyright (c) 2016 by Contributors
 * \file broadcast_reduce_op.cu
 * \brief GPU Implementation of broadcast and reduce functions.
 */
#include "./broadcast_reduce_op.h"
#include <mshadow/cuda/tensor_gpu-inl.cuh>
#include <limits>
#include <mshadow/base.h>
#include <vector>


namespace mxnet {
namespace op {

template<typename TD, typename TI>
__device__ void argmax2d_thread_reduce(TD* data, TD* shared_data, TI* shared_index, TD lowest, int bsize, int hsize) { 
  if (blockIdx.x > bsize) return;

  TD * batch_data = data + blockIdx.x * hsize;
  TD  thread_local_max_data = lowest;
  TI  thread_local_max_idx = 0;

  for (int i = threadIdx.x ; i < hsize; i += blockDim.x) {
    if (batch_data[i] > thread_local_max_data) { 
      thread_local_max_data = batch_data[i];
      thread_local_max_idx = i;
    }
  }
  
  shared_data[threadIdx.x] = thread_local_max_data;
  shared_index[threadIdx.x] = thread_local_max_idx;
}


template<typename TD, typename TI>
__device__ void argmax2d_block_reduce(TD* shared_data, TI* shared_index) {
  int left, right;
  int threads = blockDim.x / 2;
  for (int stride = 1; stride < blockDim.x; stride *= 2, threads /=2 ) {
    if (threadIdx.x < threads) {
      left = threadIdx.x * (stride * 2);
      right = left + stride;
      if (shared_data[left] < shared_data[right]) {
        shared_data[left] = shared_data[right];
        shared_index[left] = shared_index[right];
      }
    }
    __syncthreads();
  }
}


template<typename TD, typename TI>
__launch_bounds__(mshadow::cuda::kMaxThreadsPerBlock)
__global__ void argmax2d_kernel(TD* data, TI* index_out, TD lowest, int bsize, int hsize) { 
  // shared memory  
  extern __shared__ int s[]; 
  TD* shared_data = reinterpret_cast<TD*>(s);
  TI* shared_index = reinterpret_cast<TI*>(shared_data + blockDim.x);

  argmax2d_thread_reduce(data, shared_data, shared_index, lowest, bsize, hsize);
  __syncthreads();
  // for each block do block reduce
  argmax2d_block_reduce(shared_data, shared_index);

  // write to global memory
  if (threadIdx.x == 0) {
    index_out[blockIdx.x] = shared_index[0];
  }
}


template<typename TD, typename TI>
void Argmax2DComputeImpl(mshadow::Stream<gpu> *s, TD* data_in, TI* index_out, TD lowest, int bsize, int hsize){
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  int numBlocks = bsize;
  int threadsPerBlock = 512;
  size_t dev_sm_bytes = threadsPerBlock * (sizeof(TD) + sizeof(TI));
  argmax2d_kernel<<<numBlocks, threadsPerBlock, dev_sm_bytes, stream>>>(data_in, index_out, 
      lowest, bsize, hsize);
}



void Argmax2DOpCUDA(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req, 
                    const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  //CHECK_EQ(outputs[0].type_flag_, mshadow::kInt32) << " index should be type of int32";
  CHECK_EQ(outputs[0].type_flag_, mshadow::kFloat32) << " index should be type of float32";

  CHECK_EQ(inputs[0].shape_.ndim(), 2) << " Currently only support 2d reduction.";

  int bsize = inputs[0].shape_[0];
  int hsize = inputs[0].shape_[1];

  Stream<gpu> *s = ctx.get_stream<gpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, IType, {  // output index type
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, { // input data type
      Argmax2DComputeImpl<DType, IType>(s, 
                          inputs[0].dptr<DType>(),
                          outputs[0].dptr<IType>(),
                          std::numeric_limits<DType>::lowest(),
                          bsize,
                          hsize);
    });
  });
}


NNVM_REGISTER_OP(argmax)
.set_attr<FCompute>("FCompute<gpu>", SearchAxisCompute<gpu, mshadow::red::maximum>);

NNVM_REGISTER_OP(argmin)
.set_attr<FCompute>("FCompute<gpu>", SearchAxisCompute<gpu, mshadow::red::minimum>);

// Legacy support
NNVM_REGISTER_OP(argmax_channel)
.set_attr<FCompute>("FCompute<gpu>", SearchAxisCompute<gpu, mshadow::red::maximum>);

NNVM_REGISTER_OP(pick)
.set_attr<FCompute>("FCompute<gpu>", PickOpForward<gpu>);


NNVM_REGISTER_OP(_backward_pick)
.set_attr<FCompute>("FCompute<gpu>", PickOpBackward<gpu>);

// (pin)
NNVM_REGISTER_OP(argmax2d)
.set_attr<FCompute>("FCompute<gpu>", Argmax2DOpCUDA);



}  // namespace op
}  // namespace mxnet
