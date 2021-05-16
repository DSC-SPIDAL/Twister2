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
package edu.iu.dsc.tws.dl.utils.graph;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import com.intel.analytics.bigdl.mkl.Memory;

import edu.iu.dsc.tws.dl.data.Activity;
import edu.iu.dsc.tws.dl.data.Table;
import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.graph.Graph;
import edu.iu.dsc.tws.dl.graph.Node;
import edu.iu.dsc.tws.dl.module.AbstractModule;
import edu.iu.dsc.tws.dl.module.mkldnn.DnnGraph;
import edu.iu.dsc.tws.dl.module.mkldnn.MemoryData;
import edu.iu.dsc.tws.dl.module.mkldnn.MklDnnRuntime;
import edu.iu.dsc.tws.dl.module.mkldnn.Phase;
import edu.iu.dsc.tws.dl.module.mkldnn.memory.data.HeapData;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;

public class IRGraph extends AbstractModule implements Serializable {

  protected ArrayList<Node<IRElement>> inputs;
  protected ArrayList<Node<IRElement>> outputs;
  protected TensorArrayPair variables;
  protected boolean generateBackward;
  protected List<Integer> inputFormats; //Seq(Memory.Format.nchw),
  protected List<Integer> outputFormats; //Seq(Memory.Format.nc))
  private boolean initPrim = false;

  private Graph graph = null;

  public IRGraph(ArrayList<Node<IRElement>> inputsIR, ArrayList<Node<IRElement>> outputsIR,
                 TensorArrayPair vars, boolean genBackward, List<Integer> inFormats,
                 List<Integer> outFormats) {
    this.inputs = inputsIR;
    this.outputs = outputsIR;
    this.variables = vars;
    this.generateBackward = genBackward;
    this.inputFormats = inFormats;
    this.outputFormats = outFormats;
  }

  public boolean isBuild() {
    return graph != null;
  }

  @Override
  public Activity updateOutput(Activity input) {
    if (graph == null) {
      throw new UnsupportedOperationException("forward not supported, Please build graph first");
    }
    if (graph instanceof DnnGraph) {
      // if using multi MKL-DNN model, we just use current thread directly
      // because it's in sequential mode of MKL and MKL-DNN
      initPrimitives(input);
      graph.updateOutput(input);
    } else {
      graph.updateOutput(input);
    }
    output = graph.output;
    return output;
  }

  @Override
  public Activity updateGradInput(Activity input, Activity gradOutput) {
    if (graph == null) {
      throw new UnsupportedOperationException("backward not supported, Please build graph first");
    }

    graph.updateGradInput(input, gradOutput);
    gradInput = graph.gradInput;
    return gradInput;
  }

  @Override
  public void accGradParameters(Activity input, Activity gradOutput) {
    if (graph == null) {
      throw new UnsupportedOperationException("backward not supported, Please build graph first");
    }
    graph.accGradParameters(input, gradOutput);
  }

  public IRGraph build() {
    graph = new IRConverter(this).toGraph();
    return this;
  }

  @Override
  public TensorArrayPair parameters() {
    return graph.parameters();
  }

  @Override
  public List<AbstractModule> getModules() {
    return graph.modules;
  }

  @Override
  public void setModules(List list) {
    this.graph.modules = list;
  }

  @Override
  public Tensor[] getExtraParameter() {
    return graph.getExtraParameter();
  }

  @Override
  public AbstractModule training() {
    train = true;
    graph.training();
    return this;
  }

  @Override
  public AbstractModule evaluate() {
    train = false;
    graph.evaluate();
    return this;
  }


  @Override
  public Object[] getTimes() {
    return graph.getTimes();

  }

  @Override
  public void resetTimes() {
    graph.resetTimes();
  }

  private void initPrimitives(Activity input) {
    if (!initPrim && graph instanceof DnnGraph) {
      MemoryData[] inputMemory = new MemoryData[inputFormats.size()];
      if (input instanceof Tensor) {
        // todo: handle for 3 dimensions, expand 3 dims to 4 dims
        int[] size = input.toTensor().size();
        int[] sizeNew;
        if (size.length == 3 && inputFormats.get(0) != Memory.Format.ntc
            && inputFormats.get(0) != Memory.Format.tnc) {
          sizeNew = new int[]{size[0], 1, size[1], size[2]};
        } else if (inputFormats.get(0) == Memory.Format.nhwc) {
          // always use NCHW to create heap data
          sizeNew = new int[]{size[0], size[3], size[1], size[2]};

        } else {
          sizeNew = size;
        }
        inputMemory[0] = new HeapData(sizeNew, inputFormats.get(0));
      } else {
        Table tensors = input.toTable();
        Util.require(tensors.length() == inputFormats.size(), "table input length "
            + "${tensors.length()} should be the same with inputFormats"
            + " length ${inputFormats.size()}");
        tensors.forEach((tt1, tt2) -> {
          Util.require(tt2 instanceof Tensor,
              "Only support input with tensor type, table not supported");
          int t1 = (int) tt1; // starts from 1
          Tensor t2 = (Tensor) tt2;
          if (inputFormats.get(t1 - 1) == Memory.Format.nhwc) {
            int[] sizeNew = new int[]{t2.size(1), t2.size(4), t2.size(2),
                t2.size(3)};
            inputMemory[t1 - 1] = new HeapData(sizeNew, inputFormats.get(t1 - 1));
          } else {
            inputMemory[t1 - 1] = new HeapData(t2.size(), inputFormats.get(t1 - 1));
          }
        });
      }
      DnnGraph dnnGraph = (DnnGraph) graph;
      Phase phase;
      if (dnnGraph.isTraining()) {
        phase = Phase.TRAINNING;
      } else {
        phase = Phase.INFERENCE;
      }
      dnnGraph.setRuntime(new MklDnnRuntime());
      dnnGraph.initFwdPrimitives(inputMemory, phase);
      if (dnnGraph.isTraining()) {
        dnnGraph.initBwdPrimitives(dnnGraph.outputFormats(), phase);
        dnnGraph.initGradWPrimitives(dnnGraph.outputFormats(), phase);
      }
      initPrim = true;
    }
  }

  IRGraph setQuantize(boolean value) {
    Util.require(graph != null, "you should build the graph first");
    if (graph instanceof DnnGraph) {
      ((DnnGraph) graph).setQuantize(value);
    }
    return this;
  }

  @Override
  public void release() {
    if (graph instanceof DnnGraph) {
      //TODO threading if needed
      graph.release();
    }
  }

  @Override
  public void reset() {

  }
}
