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
package edu.iu.dsc.tws.dl.module.mkldnn;

import com.intel.analytics.bigdl.mkl.Memory;

import edu.iu.dsc.tws.dl.data.Activity;
import edu.iu.dsc.tws.dl.module.mkldnn.memory.data.HeapData;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.MemoryDataArrayPair;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;

@SuppressWarnings("MemberName")
public class Output extends MklDnnLayer {

  private int _outputLayOut;
  private int _gradOutputLayout;

  public Output(int outputLayout, int gradOutputLayout) {
    super();
    this._outputLayOut = outputLayout;
    if (gradOutputLayout == -1) {
      this._gradOutputLayout = outputLayout;
    } else {
      this._gradOutputLayout = gradOutputLayout;
    }
  }

  public Output(int outputLayout) {
    this(outputLayout, -1);
  }

  @Override
  public TensorArrayPair parameters() {
    return null;
  }

  @Override
  public void reset() {

  }

  @Override
  public Activity updateOutput(Activity input) {
    output = input;
    return output;
  }

  @Override
  public Activity updateGradInput(Activity input, Activity gradOutput) {
    gradInput = gradOutput;
    return gradInput;
  }

  @Override
  public MemoryDataArrayPair initFwdPrimitives(MemoryData[] inputs, Phase phase) {
    Util.require(inputs.length == 1, "Only accept one tensor as input");
    Util.require(inputs[0].shape().length == 4 || inputs[0].shape().length == 2
            || inputs[0].shape().length == 3,
        "Only support input with 2 or 3 or 4 dimentions,"
            + " but get ${inputs[0].shape().length}");

    int[] outputShape = getShape(inputs[0].layout(), inputs[0].shape(), _outputLayOut);
    // remind: output memory storage should be heapData
    _outputFormats = new MemoryData[]{new HeapData(outputShape, _outputLayOut)};
    _inputFormats = _outputFormats;

    return new MemoryDataArrayPair(_inputFormats, _outputFormats);
  }

  @Override
  public MemoryDataArrayPair initBwdPrimitives(MemoryData[] grad, Phase phase) {
    Util.require(grad.length == 1, "Only accept one tensor as input");
    Util.require(grad[0].shape().length == 4 || grad[0].shape().length == 2
            || grad[0].shape().length == 3,
        "Only support gradOutput with 2 or 3 or 4 dimentions,"
            + " but get ${grad[0].shape().length}");

    int[] outputShape = getShape(grad[0].layout(), grad[0].shape(), _gradOutputLayout);

    _gradInputFormats = new MemoryData[]{new HeapData(outputShape, _gradOutputLayout)};
    _gradOutputFormats = _gradInputFormats;
    _gradOutputFormatsForWeight = _gradOutputFormats;

    return new MemoryDataArrayPair(_gradInputFormats, _gradOutputFormats);
  }

  private int[] getShape(int inLayout, int[] inShape, int outLayout) {
    int[] outputShape;
    if (outLayout == Memory.Format.tnc && inLayout == Memory.Format.ntc) {
      // ntc -> tnc
      outputShape = new int[]{inShape[1], inShape[0], inShape[2]};
    } else if (outLayout == Memory.Format.ntc && inLayout == Memory.Format.tnc) {
      // tnc -> ntc
      outputShape = new int[]{inShape[1], inShape[0], inShape[2]};
    } else {
      outputShape = inShape;
    }
    return outputShape;
  }
}
