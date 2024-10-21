/*
 * Copyright (C) 2021-2022 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Authors: Davide Nadalini, Leonardo Ravaglia
*/ 

#include "pulp_train_defines.h"

/**
 * 2D Convolution layer configuration structure
 */

/**
 * @brief Structure for 2D Convolution Training in FP32
 * @param input input feature maps for the conv2d layer
 * @param coeff weight matrix 
 * @param output output feature maps for the conv2d layer 
 * @param Lpad left padding
 * @param Rpad right padding
 * @param Upad upper padding
 * @param Dpad lower padding
 * @param stride_w stride in input width
 * @param stride_h stride in input height
 * @param i2c_buffer pointer to the im2col buffer
 * @param bt_buffer pointer to the blocktranspose buffer (to compute input gradients)
 * @param skip_in_grad skips the computation of the input grad (1st DNN layer)
 * @param HWC tells the 2D Convolution if the input/output tensor is in CHW layout (HWC=0) or HWC format (HWC=1)
 * @param opt_matmul_type_fw number of the optimizer matmul to be chosen by the mm_manager for the forward primitive (see mm_manager_list.txt)
 * @param opt_matmul_type_wg number of the optimizer matmul to be chosen by the mm_manager for the weight gradient primitive (see mm_manager_list.txt)
 * @param opt_matmul_type_ig number of the optimizer matmul to be chosen by the mm_manager for the input gradient primitive (see mm_manager_list.txt)
 * @param USE_IM2COL if set to 0, the convd kernel calls for the naive implementation, if set to 1 for the im2col+matmul optimized execution
 * @param USE_DMA_IM2COL in case the primitive uses IM2COL + MM, select if to perform im2col using DMA-managed transfers from L2 to L1 (input and output gradient tensors need to be stored in L2, im2col_buffer in L1)
 */
struct Conv2D_args_fp16 {
	struct blob_fp16 * input; 
	struct blob_fp16 * coeff;
	struct blob_fp16 * output; 
	int Lpad;
	int Rpad;
	int Upad;
	int Dpad;
	int stride_h;
	int stride_w;
	fp16 * i2c_buffer;
	fp16 * bt_buffer;
	int skip_in_grad;
	int HWC;
	int opt_matmul_type_fw;
	int opt_matmul_type_wg;
	int opt_matmul_type_ig;
	int USE_IM2COL;
	int USE_DMA_IM2COL;
};



/**
 * Convolutional layer training functions, grouped into FW and BW
 */


// FORWARD FUNCTIONS

/**
 * @brief Forward pass function, forked on PULP cluster.
 * @param input input feauture maps for the conv2d layer
 * @param coeff weight matrix 
 * @param output output feature maps for the conv2d layer
 * @param Lpad left padding
 * @param Rpad right padding
 * @param Upad upper padding
 * @param Dpad lower padding
 * @param stride_w stride in input width
 * @param stride_h stride in input height
 * @param i2c_buffer pointer to the im2col buffer
 * @param HWC tells the 2D Convolution if the input tensor is in CHW layout (HWC=0) or HWC format (HWC=1)
 * @param opt_matmul_type_fw number of the optimizer matmul to be chosen by the mm_manager (see mm_manager_list.txt)
 * @param USE_IM2COL if set to 0, the convd kernel calls for the naive implementation, if set to 1 for the im2col+matmul optimized execution
 * @param USE_DMA_IM2COL in case the primitive uses IM2COL + MM, select if to perform im2col using DMA-managed transfers from L2 to L1 (input tensor needs to be stored in L2, im2col_buffer in L1)
 */
void pulp_conv2d_fp16_fw_cl( void * Conv2D_args_fp16 );


// BACKWARD FUNCTIONS

/**
 * @brief Backward pass function, which internally calls both weight gradient and input gradient calculation
 * @param input input feauture maps for the conv2d layer
 * @param coeff weight matrix 
 * @param output output feature maps for the conv2d layer 
 * @param Lpad left padding
 * @param Rpad right padding
 * @param Upad upper padding
 * @param Dpad lower padding
 * @param stride_w stride in input width
 * @param stride_h stride in input height
 * @param i2c_buffer pointer to the im2col buffer
 * @param bt_buffer pointer to the blocktranspose buffer (to compute input gradients)
 * @param skip_in_grad skips the computation of the input grad (1st DNN layer)
 * @param HWC tells the 2D Convolution if the input/output tensor is in CHW layout (HWC=0) or HWC format (HWC=1)
 * @param opt_matmul_type_wg number of the optimizer matmul to be chosen by the mm_manager for the weight gradient primitive (see mm_manager_list.txt)
 * @param opt_matmul_type_ig number of the optimizer matmul to be chosen by the mm_manager for the input gradient primitive (see mm_manager_list.txt)
 * @param USE_IM2COL if set to 0, the convd kernel calls for the naive implementation, if set to 1 for the im2col+matmul optimized execution
 * @param USE_DMA_IM2COL in case the primitive uses IM2COL + MM, select if to perform im2col using DMA-managed transfers from L2 to L1 (input and output gradient tensors need to be stored in L2, im2col_buffer in L1)
 */
void pulp_conv2d_fp16_bw_cl( void * Conv2D_args_fp16 );

/**
 * @brief Backward pass function which computes weight's gradient only
 * @param input input feauture maps for the conv2d layer
 * @param coeff weight matrix 
 * @param output output feature maps for the conv2d layer 
 * @param Lpad left padding
 * @param Rpad right padding
 * @param Upad upper padding
 * @param Dpad lower padding
 * @param stride_w stride in input width
 * @param stride_h stride in input height
 * @param i2c_buffer pointer to the im2col buffer
 * @param HWC tells the 2D Convolution if the input tensor is in CHW layout (HWC=0) or HWC format (HWC=1)
 * @param opt_matmul_type_wg number of the optimizer matmul to be chosen by the mm_manager (see mm_manager_list.txt)
 * @param USE_IM2COL if set to 0, the convd kernel calls for the naive implementation, if set to 1 for the im2col+matmul optimized execution
 * @param USE_DMA_IM2COL in case the primitive uses IM2COL + MM, select if to perform im2col using DMA-managed transfers from L2 to L1 (input tensor needs to be stored in L2, im2col_buffer in L1)
 */
void pulp_conv2d_fp16_bw_param_grads_cl( void * Conv2D_args_fp16 );

/**
 * @brief Backward pass function which computes input's gradient only
 * @param input input feauture maps for the conv2d layer
 * @param coeff weight matrix 
 * @param output output feature maps for the conv2d layer 
 * @param Lpad left padding
 * @param Rpad right padding
 * @param Upad upper padding
 * @param Dpad lower padding
 * @param stride_w stride in input width
 * @param stride_h stride in input height
 * @param i2c_buffer pointer to the im2col buffer
 * @param bt_buffer pointer to the blocktranspose buffer (to reshape the weights for the in grad step)
 * @param HWC tells the 2D Convolution if the output tensor is in CHW layout (HWC=0) or HWC format (HWC=1)
 * @param opt_matmul_type_ig number of the optimizer matmul to be chosen by the mm_manager (see mm_manager_list.txt)
 * @param USE_IM2COL if set to 0, the convd kernel calls for the naive implementation, if set to 1 for the im2col+matmul optimized execution
 * @param USE_DMA_IM2COL in case the primitive uses IM2COL + MM, select if to perform im2col using DMA-managed transfers from L2 to L1 (output gradient tensor needs to be stored in L2, im2col_buffer in L1)
 */
void pulp_conv2d_fp16_bw_input_grads_cl( void * Conv2D_args_fp16 );
