#!/usr/bin/python
# -*- coding: utf-8 -*-

import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# mod = SourceModule("""
# __global__ void soft_thresholding(float* precalc, const int N, float nsigma, float mu_s, float tau_s, float* aux_beta, float* out_beta_next)
# {
#     int idx = threadIdx.x + blockIdx.x * blockDim.x;
#
#     if (idx < N)
#     {
#         /* value = (precalc / nsigma) + ((1.0 - mu_s) * aux_beta) */
#
#         float left = (precalc[idx] / nsigma);
#         float right = (1 - mu_s) * aux_beta[idx];
#         float value = right + left;
#
#         float newval = fabsf(value) - (tau_s);
#         out_beta_next[idx] = (!signbit(value)*2 - 1) * (!signbit(newval))*newval;
#     }
# }
mod = SourceModule("""
__global__ void soft_thresholdingFISTA(float* precalc, const int N, float nsigma, float mu_s, float tau_s, float* beta, float* aux_beta, float* beta_next, float t, float* t_next, float* beta_diff)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N)
    {
        // ################ SOFT THRESHOLDING ################
        /* value = (precalc / nsigma) + ((1.0 - mu_s) * aux_beta) */
        float left = (precalc[idx] / nsigma);
        float right = (1 - mu_s) * aux_beta[idx];
        float value = right + left;

        /* beta_next = np.sign(value) * np.clip(np.abs(value) - tau_s, 0, np.inf) */
        float newval = fabsf(value) - (tau_s);
        beta_next[idx] = (!signbit(value)*2 - 1) * (!signbit(newval))*newval;

        // ################ FISTA ################
        /* beta_diff = (beta_next - beta) */
        beta_diff[idx] = beta_next[idx] - beta[idx];

        /* t_next = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t*t)) */
        t_next[0] = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t*t));

        /* aux_beta = beta_next + ((t - 1.0)/t_next)*beta_diff */
        aux_beta[idx] = beta_next[idx] + ((t - 1.0)/t_next[0])*beta_diff[idx];
    }
}
""")

# /* beta_next = np.sign(value) * np.clip(np.abs(value) - tau_s, 0, np.inf) */
# //float sign = 2 * !signbit(value) - 1;
# //float clip_arg = fabsf(value) - tau_s;
# //if(clip_arg > 0)
# //     out_beta_next[idx] = clip_arg * sign;
# // else
# //     out_beta_next[idx] = 0;


#
# __global__ void soft_thresh_step1(float* precalc, float* nsigma, float* mu_s, float* aux_beta, float* out_value)
#  {
#    int idx = threadIdx.x;
#
#    float left = precalc[idx] / nsigma;
#    float right = (1 - mu_s) * aux_beta[idx];
#
#    out_value[idx] = right + left;
#  }
#
#
# __global__ void soft_thresh_step2(float* value, float* tau_s, float* out_beta_next)
#  {
#    int idx = threadIdx.x;
#
#    float sign = 2 * !signbit(value[idx]) - 1;
#    float clip_arg = fabsf(value[idx]) - tau_s;
#
#    if(clip_arg > 0)
#         out_beta_next[idx] = clip_arg * sign;
#     else
#         out_beta_next[idx] = 0;
#  }
