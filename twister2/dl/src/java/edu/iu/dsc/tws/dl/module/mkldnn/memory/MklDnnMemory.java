//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
package edu.iu.dsc.tws.dl.module.mkldnn.memory;

import com.intel.analytics.bigdl.mkl.MklDnn;

import edu.iu.dsc.tws.dl.module.mkldnn.MemoryOwner;

@SuppressWarnings({"LocalVariableName", "ParameterName"})
public final class MklDnnMemory {

  private MklDnnMemory() {
  }

  public static long MemoryDescInit(int ndims, int[] dims, int dataType, int dataFormat,
                                    MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.MemoryDescInit(ndims, dims, dataType, dataFormat), owner).getPtr();
  }

  public static long EltwiseForwardDescInit(int propKind, int algKind, long srcDesc, float alpha,
                                            float beta, MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.EltwiseForwardDescInit(propKind, algKind, srcDesc, alpha, beta), owner).getPtr();
  }

  public static long EltwiseBackwardDescInit(int algKind, long diffDataDesc, long dataDesc,
                                             float alpha, float beta, MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.EltwiseBackwardDescInit(algKind, diffDataDesc, dataDesc,
            alpha, beta), owner).getPtr();
  }

  public static long LinearForwardDescInit(int propKind, long srcMemDesc, long weightMemDesc,
                                           long biasMemDesc, long dstMemDesc,
                                           MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.LinearForwardDescInit(propKind, srcMemDesc, weightMemDesc, biasMemDesc,
            dstMemDesc), owner).getPtr();
  }

  public static long LinearBackwardDataDescInit(long diffSrcMemDesc, long weightMemDesc,
                                                long diffDstMemDesc, MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.LinearBackwardDataDescInit(diffSrcMemDesc, weightMemDesc, diffDstMemDesc),
        owner).getPtr();
  }

  public static long LinearBackwardWeightsDescInit(long srcMemDesc, long diffWeightMemDesc,
                                                   long diffBiasMemDesc, long diffDstMemDesc,
                                                   MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.LinearBackwardWeightsDescInit(srcMemDesc,
            diffWeightMemDesc, diffBiasMemDesc, diffDstMemDesc), owner).getPtr();
  }

  public static long BatchNormForwardDescInit(int propKind, long srcMemDesc, float epsilon,
                                              long flags, MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.BatchNormForwardDescInit(propKind, srcMemDesc, epsilon, flags), owner).getPtr();
  }

  public static long BatchNormBackwardDescInit(int prop_kind, long diffDstMemDesc, long srcMemDesc,
                                               float epsilon, long flags, MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.BatchNormBackwardDescInit(prop_kind, diffDstMemDesc, srcMemDesc, epsilon,
            flags), owner).getPtr();
  }

  public static long SoftMaxForwardDescInit(int prop_kind, long dataDesc, int axis,
                                            MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.SoftMaxForwardDescInit(prop_kind, dataDesc, axis), owner).getPtr();
  }

  public static long SoftMaxBackwardDescInit(int propKind, long diffDesc, long dstDesc,
                                             int axis, MemoryOwner owner) {
    return new MklMemoryDescInit(MklDnn.SoftMaxBackwardDescInit(diffDesc, dstDesc, axis),
        owner).getPtr();
  }

  public static long ConvForwardDescInit(int prop_kind, int alg_kind, long src_desc,
                                         long weights_desc, long bias_desc, long dst_desc,
                                         int[] strides, int[] padding_l, int[] padding_r,
                                         int padding_kind, MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.ConvForwardDescInit(prop_kind, alg_kind, src_desc, weights_desc,
            bias_desc, dst_desc, strides, padding_l,
            padding_r, padding_kind), owner).getPtr();
  }

  public static long DilatedConvForwardDescInit(int prop_kind, int alg_kind, long src_desc,
                                                long weights_desc, long bias_desc,
                                                long dst_desc, int[] strides,
                                                int[] dilates, int[] padding_l, int[] padding_r,
                                                int padding_kind, MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.DilatedConvForwardDescInit(prop_kind, alg_kind, src_desc,
            weights_desc, bias_desc, dst_desc, strides,
            dilates, padding_l, padding_r, padding_kind), owner).getPtr();
  }

  public static long ConvBackwardWeightsDescInit(int alg_kind, long src_desc,
                                                 long diff_weights_desc,
                                                 long diff_bias_desc, long diff_dst_desc,
                                                 int[] strides, int[] padding_l, int[] padding_r,
                                                 int padding_kind, MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.ConvBackwardWeightsDescInit(alg_kind, src_desc, diff_weights_desc,
            diff_bias_desc, diff_dst_desc, strides, padding_l,
            padding_r, padding_kind), owner).getPtr();
  }

  public static long DilatedConvBackwardWeightsDescInit(int alg_kind, long src_desc,
                                                        long diff_weights_desc,
                                                        long diff_bias_desc, long diff_dst_desc,
                                                        int[] strides, int[] dilates,
                                                        int[] padding_l, int[] padding_r,
                                                        int padding_kind, MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.DilatedConvBackwardWeightsDescInit(alg_kind, src_desc, diff_weights_desc,
            diff_bias_desc, diff_dst_desc, strides, dilates,
            padding_l, padding_r, padding_kind), owner).getPtr();
  }

  public static long ConvBackwardDataDescInit(int alg_kind, long diff_src_desc, long weights_desc,
                                              long diff_dst_desc, int[] strides,
                                              int[] padding_l, int[] padding_r,
                                              int padding_kind, MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.ConvBackwardDataDescInit(alg_kind, diff_src_desc, weights_desc,
            diff_dst_desc, strides, padding_l, padding_r,
            padding_kind), owner).getPtr();
  }

  public static long DilatedConvBackwardDataDescInit(int alg_kind, long diff_src_desc,
                                                     long weights_desc,
                                                     long diff_dst_desc, int[] strides,
                                                     int[] padding_l, int[] dilates,
                                                     int[] padding_r, int padding_kind,
                                                     MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.DilatedConvBackwardDataDescInit(alg_kind, diff_src_desc, weights_desc,
            diff_dst_desc, strides, padding_l, dilates,
            padding_r, padding_kind), owner).getPtr();
  }

  public static long PoolingForwardDescInit(int prop_kind, int alg_kind, long src_desc,
                                            long dst_desc,
                                            int[] strides, int[] kernel, int[] padding_l,
                                            int[] padding_r,
                                            int padding_kind, MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.PoolingForwardDescInit(prop_kind, alg_kind, src_desc, dst_desc,
            strides, kernel, padding_l, padding_r,
            padding_kind), owner).getPtr();
  }

  public static long PoolingBackwardDescInit(int alg_kind, long diff_src_desc, long diff_dst_desc,
                                             int[] strides, int[] kernel, int[] padding_l,
                                             int[] padding_r, int padding_kind, MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.PoolingBackwardDescInit(alg_kind, diff_src_desc, diff_dst_desc,
            strides, kernel, padding_l, padding_r,
            padding_kind), owner).getPtr();
  }

  public static long LRNForwardDescInit(int prop_kind, int alg_kind, long data_desc, int local_size,
                                        float alpha, float beta, float k, MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.LRNForwardDescInit(prop_kind, alg_kind, data_desc, local_size,
            alpha, beta, k), owner).getPtr();
  }

  public static long LRNBackwardDescInit(int alg_kind, long diff_data_desc, long data_desc,
                                         int local_size,
                                         float alpha, float beta, float k, MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.LRNBackwardDescInit(alg_kind, diff_data_desc, data_desc,
            local_size, alpha, beta, k), owner).getPtr();
  }

  public static long RNNCellDescInit(int kind, int f, int flags, float alpha, float clipping,
                                     MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.RNNCellDescInit(kind, f, flags, alpha, clipping), owner).getPtr();
  }

  public static long RNNForwardDescInit(int prop_kind, long rnn_cell_desc, int direction,
                                        long src_layer_desc, long src_iter_desc,
                                        long weights_layer_desc, long weights_iter_desc,
                                        long bias_desc, long dst_layer_desc, long dst_iter_desc,
                                        MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.RNNForwardDescInit(prop_kind, rnn_cell_desc, direction,
            src_layer_desc, src_iter_desc, weights_layer_desc, weights_iter_desc,
            bias_desc, dst_layer_desc, dst_iter_desc), owner).getPtr();
  }

  public static long RNNBackwardDescInit(int prop_kind, long rnn_cell_desc, int direction,
                                         long src_layer_desc, long src_iter_desc,
                                         long weights_layer_desc, long weights_iter_desc,
                                         long bias_desc, long dst_layer_desc, long dst_iter_desc,
                                         long diff_src_layer_desc,
                                         long diff_src_iter_desc, long diff_weights_layer_desc,
                                         long diff_weights_iter_desc,
                                         long diff_bias_desc, long diff_dst_layer_desc,
                                         long diff_dst_iter_desc, MemoryOwner owner) {
    return new MklMemoryDescInit(
        MklDnn.RNNBackwardDescInit(prop_kind, rnn_cell_desc, direction,
            src_layer_desc, src_iter_desc, weights_layer_desc, weights_iter_desc,
            bias_desc, dst_layer_desc, dst_iter_desc, diff_src_layer_desc,
            diff_src_iter_desc, diff_weights_layer_desc, diff_weights_iter_desc,
            diff_bias_desc, diff_dst_layer_desc, diff_dst_iter_desc), owner).getPtr();
  }

  public static long ReorderPrimitiveDescCreate(long input, long output, MemoryOwner owner) {
    return new MklMemoryPrimitiveDesc(
        MklDnn.ReorderPrimitiveDescCreate(input, output), owner).getPtr();
  }

  public static long ReorderPrimitiveDescCreateV2(long input, long output, long attr,
                                                  MemoryOwner owner) {
    return new MklMemoryPrimitiveDesc(
        MklDnn.ReorderPrimitiveDescCreateV2(input, output, attr), owner).getPtr();
  }

  public static long PrimitiveCreate0(long desc, MemoryOwner owner) {
    return new MklMemoryPrimitive(
        MklDnn.PrimitiveCreate0(desc), owner).getPtr();
  }

  public static long PrimitiveCreate2(long desc, long[] inputs, int[] indexes, int inputLen,
                                      long[] outputs, int outputLen, MemoryOwner owner) {
    return new MklMemoryPrimitive(
        MklDnn.PrimitiveCreate2(desc, inputs, indexes, inputLen,
            outputs, outputLen), owner).getPtr();
  }

  public static long PrimitiveDescCreate(long opDesc, long engine, long hingForwardPrimitiveDesc,
                                         MemoryOwner owner) {
    return new MklMemoryPrimitiveDesc(MklDnn.PrimitiveDescCreate(opDesc, engine,
        hingForwardPrimitiveDesc), owner).getPtr();
  }

  public static long PrimitiveDescCreateV2(long opDesc, long attr, long engine,
                                           long hingForwardPrimitiveDesc, MemoryOwner owner) {
    return new MklMemoryPrimitiveDesc(
        MklDnn.PrimitiveDescCreateV2(opDesc, attr, engine,
            hingForwardPrimitiveDesc), owner).getPtr();
  }

  public static long MemoryPrimitiveDescCreate(long desc, long engine, MemoryOwner owner) {
    return new MklMemoryPrimitiveDesc(
        MklDnn.MemoryPrimitiveDescCreate(desc, engine), owner).getPtr();
  }

  public static long ConcatPrimitiveDescCreate(long output_desc, int n, int concat_dimension,
                                               long[] input_pds, MemoryOwner owner) {
    return new MklMemoryPrimitiveDesc(
        MklDnn.ConcatPrimitiveDescCreate(output_desc, n, concat_dimension,
            input_pds), owner).getPtr();
  }

  public static long ViewPrimitiveDescCreate(long memory_primitive_desc, int[] dims, int[] offsets,
                                             MemoryOwner owner) {
    return new MklMemoryPrimitiveDesc(
        MklDnn.ViewPrimitiveDescCreate(memory_primitive_desc, dims, offsets), owner).getPtr();
  }

  public static long SumPrimitiveDescCreate(long output_mem_desc, int n, float[] scales,
                                            long[] input_pds, MemoryOwner owner) {
    return new MklMemoryPrimitiveDesc(
        MklDnn.SumPrimitiveDescCreate(output_mem_desc, n, scales,
            input_pds), owner).getPtr();
  }

  public static long CreateAttr(MemoryOwner owner) {
    return new MklMemoryAttr(
        MklDnn.CreateAttr(), owner).getPtr();
  }

  public static long CreatePostOps(MemoryOwner owner) {
    return new MklMemoryPostOps(
        MklDnn.CreatePostOps(), owner).getPtr();
  }
}
