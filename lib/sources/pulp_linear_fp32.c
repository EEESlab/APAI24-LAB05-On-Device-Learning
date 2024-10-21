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

#include "pulp_train_utils_fp32.h"
#include "pulp_matmul_fp32.h"
#include "pulp_linear_fp32.h"

void pulp_linear_fp32_fw_cl( void * Linear_args )
{
  struct Linear_args * FC_args = (struct Linear_args *) Linear_args;
  float *coeffData = FC_args->coeff->data;
  float *outData = FC_args->output->data;  
  float *inputData = FC_args->input->data;

  int Ci = FC_args->input->dim;
  int Co = FC_args->output->dim;

  int opt_matmul_type = FC_args->opt_matmul_type_fw;

  struct matMul_args matMul_args;

  matMul_args.A = coeffData;
  matMul_args.B = inputData;
  matMul_args.C = outData;
  matMul_args.N = Co; //FC_args->output->dim;
  matMul_args.K = Ci; //FC_args->input->dim;
  matMul_args.M = 1;
  matMul_args.trans_B = 0;

  #ifndef OPTIMIZE
  pi_cl_team_fork(NUM_CORES, mm, &matMul_args);
  #else
  struct mm_manager_args man_args;
  man_args.mm_args = &matMul_args;
  man_args.layer_type = LAYER_LINEAR;
  man_args.step_type = STEP_FW;
  man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;
  pi_cl_team_fork(NUM_CORES, mm_manager, &man_args);
  #endif

}


void pulp_linear_fp32_bw_cl( void * Linear_args )
{
  struct Linear_args * FC_args = (struct Linear_args *) Linear_args;
  int skip_in_grad = FC_args->skip_in_grad;

  pulp_linear_fp32_bw_param_grads_cl(Linear_args);
  if (skip_in_grad == 0) 
  {
    pulp_linear_fp32_bw_input_grads_cl(Linear_args); 
  }
}


void pulp_linear_fp32_bw_param_grads_cl( void * Linear_args )
{
  struct Linear_args * FC_args = (struct Linear_args *) Linear_args;
  float *coeffData = FC_args->coeff->data;
  float *inData = FC_args->input->data;
  float *outData = FC_args->output->data;
  float *coeffDiff = FC_args->coeff->diff;
  float *outDiff = FC_args->output->diff;  
  float *inDiff = FC_args->input->diff;

  int Ci = FC_args->input->dim;
  int Co = FC_args->output->dim;

  int opt_matmul_type = FC_args->opt_matmul_type_wg;

  struct matMul_args matMul_args;

  matMul_args.A = outDiff;
  matMul_args.B = inData;
  matMul_args.C = coeffDiff;
  /**
   * EXERCISE 3 - SIZE OF MATRICES (WEIGHT GRAD)
  */
  // COMPLETE THE MATRIX SIZES
  matMul_args.N = /* YOUR CODE HERE, REMOVE 0; */ 0; 
  matMul_args.K = /* YOUR CODE HERE, REMOVE 0; */ 0; 
  matMul_args.M = /* YOUR CODE HERE, REMOVE 0; */ 0; 
  /**
   * END OF EXERCISE 3 - SIZE OF MATRICES (WEIGHT GRAD)
  */
  matMul_args.trans_B = 0;

  #ifndef OPTIMIZE
  pi_cl_team_fork(NUM_CORES, mm, &matMul_args);
  #else
  struct mm_manager_args man_args;
  man_args.mm_args = &matMul_args;
  man_args.layer_type = LAYER_LINEAR;
  man_args.step_type = STEP_WGT_GRAD;
  man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;
  pi_cl_team_fork(NUM_CORES, mm_manager, &man_args);
  #endif

}


void pulp_linear_fp32_bw_input_grads_cl( void * Linear_args )
{
  struct Linear_args * FC_args = (struct Linear_args *) Linear_args;
  float *coeffData = FC_args->coeff->data;
  float *inData = FC_args->input->data;
  float *outData = FC_args->output->data;
  float *coeffDiff = FC_args->coeff->diff;
  float *outDiff = FC_args->output->diff;  
  float *inDiff = FC_args->input->diff;

  int Ci = FC_args->input->dim;
  int Co = FC_args->output->dim;

  int opt_matmul_type = FC_args->opt_matmul_type_ig;

  struct matMul_args matMul_args;

  matMul_args.A = outDiff;
  matMul_args.B = coeffData;
  matMul_args.C = inDiff;
  /**
   * EXERCISE 3 - SIZE OF MATRICES (INPUT GRAD)
  */
  // COMPLETE THE MATRIX SIZES
  matMul_args.N = /* YOUR CODE HERE, REMOVE 0; */ 0; 
  matMul_args.K = /* YOUR CODE HERE, REMOVE 0; */ 0; 
  matMul_args.M = /* YOUR CODE HERE, REMOVE 0; */ 0; 
  /**
   * END OF EXERCISE 3 - SIZE OF MATRICES (INPUT GRAD)
  */
  matMul_args.trans_B = 0;

  #ifndef OPTIMIZE
  pi_cl_team_fork(NUM_CORES, mm_M, &matMul_args);
  #else
  struct mm_manager_args man_args;
  man_args.mm_args = &matMul_args;
  man_args.layer_type = LAYER_LINEAR;
  man_args.step_type = STEP_IN_GRAD;
  man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;
  pi_cl_team_fork(NUM_CORES, mm_manager, &man_args);
  #endif

}
