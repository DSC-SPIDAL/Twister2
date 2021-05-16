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

import edu.iu.dsc.tws.dl.data.Activity;
import edu.iu.dsc.tws.dl.module.AbstractModule;
import edu.iu.dsc.tws.dl.module.DynamicContainer;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.MemoryDataArrayPair;

public class Sequential extends MklDnnContainer {
  @Override
  public Activity updateOutput(Activity input) {
    int i = 0;
    Activity lastOutput = input;
    while (i < mklDnnModules.length - 1) {
      lastOutput = reorderManager.infer(
          mklDnnModules[i].outputFormats(),
          mklDnnModules[i + 1].inputFormats(),
          modules.get(i).forward(lastOutput));
      i += 1;
    }

    this.output = modules.get(i).forward(lastOutput);
    return output;
  }

  @Override
  public Activity updateGradInput(Activity input, Activity gradOutput) {
    int i = modules.size() - 1;
    Activity lastGradInput = gradOutput;
    while (i > 0) {
      Activity curInput = reorderManager.infer(
          mklDnnModules[i - 1].outputFormats(),
          mklDnnModules[i].inputFormats(),
          modules.get(i - 1).output);

      lastGradInput = reorderManager.infer(
          mklDnnModules[i].gradInputFormats(),
          mklDnnModules[i - 1].gradOutputFormats(),
          modules.get(i).updateGradInput(curInput, lastGradInput)
      );
      i -= 1;
    }
    lastGradInput = modules.get(0).updateGradInput(input, lastGradInput);

    this.gradInput = lastGradInput;
    return gradInput;
  }

  @Override
  public void accGradParameters(Activity input, Activity gradOutput) {
    int i = modules.size() - 1;
    AbstractModule currentModule = modules.get(i);
    Activity lastGradInput = gradOutput;
    while (i > 0) {
      currentModule = modules.get(i);
      Activity curInput = reorderManager.infer(
          mklDnnModules[i - 1].outputFormats(),
          mklDnnModules[i].inputFormats(),
          modules.get(i - 1).output
      );
      currentModule.accGradParameters(curInput, lastGradInput);
      currentModule.asyncGradient();
      lastGradInput = reorderManager.infer(
          mklDnnModules[i].gradInputFormats(),
          mklDnnModules[i - 1].gradOutputWeightFormats(),
          modules.get(i).gradInput
      );
      i -= 1;
    }

    modules.get(i).accGradParameters(input, lastGradInput);
    modules.get(i).asyncGradient();
  }

  @Override
  public MemoryDataArrayPair initFwdPrimitives(MemoryData[] inputs, Phase phase) {
    MemoryData[] lastOutputFormats = inputs;
    MemoryData[] firstRealInputFormats = null;
    for (int i = 0; i < mklDnnModules.length; i++) {
      MklDnnModule m = mklDnnModules[i];

      //(realInputFormats, outputFormats)
      MemoryDataArrayPair realInputOutputFormats = m.initFwdPrimitives(lastOutputFormats, phase);

      for (int j = 0; j < lastOutputFormats.length; j++) {
        Util.copyMaskAndScales(lastOutputFormats[i], realInputOutputFormats.getValue0()[i]);
        reorderManager.register(lastOutputFormats[i], realInputOutputFormats.getValue0()[i]);
      }

      Util.copyMaskAndScales(realInputOutputFormats.getValue0(),
          realInputOutputFormats.getValue1());
      if (i == 0) {
        firstRealInputFormats = realInputOutputFormats.getValue0();
      }
      lastOutputFormats = realInputOutputFormats.getValue1();
    }
    return new MemoryDataArrayPair(firstRealInputFormats, lastOutputFormats);
  }

  @Override
  public MemoryDataArrayPair initBwdPrimitives(MemoryData[] grad, Phase phase) {
    MemoryData[] lastGradInputFormats = grad;
    MemoryData[] firstRealGradOutputFormats = null;
    for (int i = mklDnnModules.length - 1; i == 0; i--) {
      MklDnnModule m = mklDnnModules[i];
      //(realGradOutput, gradInputFomrats);
      MemoryDataArrayPair realInputOutputFormats = m.initBwdPrimitives(lastGradInputFormats, phase);
      for (int j = 0; j < lastGradInputFormats.length; j++) {
        reorderManager.register(lastGradInputFormats[i], realInputOutputFormats.getValue0()[i]);
      }
      if (i == mklDnnModules.length - 1) {
        firstRealGradOutputFormats = realInputOutputFormats.getValue0();
      }
      lastGradInputFormats = realInputOutputFormats.getValue1();
    }

    return new MemoryDataArrayPair(firstRealGradOutputFormats, lastGradInputFormats);
  }

  @Override
  public MemoryData[] initGradWPrimitives(MemoryData[] grad, Phase phase) {
    MemoryData[] lastGradInputFormats = grad;
    MemoryData[] firstRealGradOutputFormats = null;
    for (int i = mklDnnModules.length - 1; i == 0; i--) {
      MklDnnModule m = mklDnnModules[i];
      MemoryData[] realGradOutput = m.initGradWPrimitives(lastGradInputFormats, phase);
      for (int j = 0; j < lastGradInputFormats.length; j++) {
        reorderManager.register(lastGradInputFormats[i], realGradOutput[i]);
      }
      if (i == mklDnnModules.length - 1) {
        firstRealGradOutputFormats = realGradOutput;
      }
      lastGradInputFormats = m.gradInputFormats();
    }
    return firstRealGradOutputFormats;
  }

  @Override
  public MemoryData[] inputFormats() {
    return ((MklDnnModule) modules.get(0)).inputFormats();
  }

  @Override
  public MemoryData[] gradInputFormats() {
    return ((MklDnnModule) modules.get(0)).gradInputFormats();
  }

  @Override
  public MemoryData[] outputFormats() {
    return ((MklDnnModule) modules.get(modules.size() - 1)).outputFormats();
  }

  @Override
  public MemoryData[] gradOutputFormats() {
    return ((MklDnnModule) modules.get(modules.size() - 1)).gradOutputFormats();
  }

  @Override
  public MemoryData[] gradOutputWeightFormats() {
    return ((MklDnnModule) modules.get(modules.size() - 1)).gradOutputWeightFormats();
  }

  @Override
  public DynamicContainer add(AbstractModule module) {
    Util.require(mklDnnModules == null, "You should not call add after compilation");
    Util.require(module instanceof MklDnnModule, "layer should be MklDnnModule");
    return super.add(module);
  }

  @Override
  protected void fusion(Phase phase) {
    modules.clear();

//    modules.appendAll(getFusedModules(phase).map {x =>
//      x.asInstanceOf[AbstractModule[Activity, Activity, Float]]
//    })
//    mklDnnModules = modules.map(_.asInstanceOf[MklDnnModule]).toArray
  }

  private boolean fuse() {
    return Boolean.parseBoolean(System.getProperty("bigdl.mkldnn.fusion", "false"));
  }

  private boolean useConvBn() {
    return fuse() || Boolean.parseBoolean(System.getProperty("bigdl.mkldnn.fusion.convbn",
        "false"));
  }

  private boolean useBnRelu() {
    return fuse() || Boolean.parseBoolean(System.getProperty("bigdl.mkldnn.fusion.bnrelu",
        "false"));
  }

  private boolean useConvRelu() {
    return fuse() || Boolean.parseBoolean(System.getProperty("bigdl.mkldnn.fusion.convrelu",
        "false"));
  }

  private boolean useConvSum() {
    return fuse() || Boolean.parseBoolean(System.getProperty("bigdl.mkldnn.fusion.convsum",
        "false"));
  }

//  private MklDnnModule[] convWithBn(MklDnnModule[] modules, Phase phase) {
//    if (fuseConvBn && phase == Phase.INFERENCE) {
//      ArrayList<MklDnnModule> newModules = new ArrayList<>();
//      SpatialBatchNormalization lastBn = null;
//
//      modules.zip(modules.drop(1) ++ Array(null)).foreach { case (f, s) =>
//        (f, s) match {
//        case (conv: SpatialConvolution, bn: SpatialBatchNormalization) =>
//          mergeConvBn(conv, bn)
//          newModules.append(conv)
//          lastBn = bn
//        case (f: MklDnnContainer, s) => f.fusion(phase); newModules.append(f)
//        case _ => if (lastBn != f) { newModules.append(f) }
//      }
//      }
//
//      newModules.toArray
//    } else {
//      modules
//    }
//  }
//
//  private MklDnnModule[] convWithReLU(MklDnnModule[] modules, Phase phase) {
//    if (fuseConvRelu) {
//      ArrayList<MklDnnModule> newModules = new ArrayList<>();
//      int lastReLU: ReLU = null
//
//      modules.zip(modules.drop(1) ++ Array(null)).foreach { case (f, s) =>
//        (f, s) match {
//        case (conv: SpatialConvolution, relu: ReLU) =>
//          newModules.append(conv)
//          conv.setReLU()
//          conv.setOutputScales(relu.getOutputScales())
//          lastReLU = relu
//        case (f: MklDnnContainer, s) =>
//          f.fusion(phase)
//          newModules.append(f)
//        case _ => if (lastReLU != f) {
//          newModules.append(f)
//        }
//      }
//      }
//
//      newModules.toArray
//    } else {
//      modules
//    }
//  }
//
//  private MklDnnModule[] bnWithReLU(MklDnnModule[] modules, Phase phase) {
//    if (fuseBnRelu) {
//      ArrayList<MklDnnModule> newModules = new ArrayList<>();
//      int lastReLU: ReLU = null
//
//      modules.zip(modules.drop(1) ++ Array(null)).foreach { case (f, s) =>
//        (f, s) match {
//        case (bn: SpatialBatchNormalization, relu: ReLU) =>
//          newModules.append(bn)
//          bn.setReLU(true)
//          lastReLU = relu
//        case (f: MklDnnContainer, s) => f.fusion(phase); newModules.append(f)
//        case _ => if (lastReLU != f) { newModules.append(f) }
//      }
//      }
//
//      newModules.toArray
//    } else {
//      modules
//    }
//  }
//
//  private MklDnnModule[] convWithSum(MklDnnModule[] modules, Phase phase) {
//    ArrayList<MklDnnModule> newModules = new ArrayList<>();
//    if (!fuseConvSum || modules.size() <= 2 || phase == TrainingPhase) {
//      newModules.appendAll(modules)
//    } else {
//      int lastConv: SpatialConvolution = null
//      int lastReLU: ReLU = null
//
//      modules.zip(modules.drop(1) ++ Array(null)).foreach {
//        case (f: ConcatTable, s: CAddTable) => val (conv, sbt) = convSum(f, s)
//          newModules.append(f)
//          lastConv = conv
//          if (sbt != null) {
//            newModules.append(sbt)
//          }
//          conv.setOutputScales(s.getOutputScales())
//        case (f: MklDnnContainer, s) => f.fusion(phase); newModules.append(f)
//        case (f: CAddTable, s: ReLU) => if (lastConv != null) {
//          lastConv.setReLU()
//          lastConv.setOutputScales(s.getOutputScales())
//          lastReLU = s
//          lastConv = null
//        } else {
//          newModules.append(f)
//        }
//        case (f, s) => if (lastReLU != f) { newModules.append(f); lastReLU = null}
//      }
//    }
//
//    newModules.toArray
//  }
//
//  private MklDnnModule[] getFusedModules(Phase phase) {
//    val f1Modules = convWithBn(mklDnnModules, phase)
//    val f2Modules = convWithReLU(f1Modules, phase)
//    val f3Modules = bnWithReLU(f2Modules, phase)
//    val f4Modules = convWithSum(f3Modules, phase)
//    f4Modules
//  }
//
//  private def mergeConvBn(conv: SpatialConvolution, bn: SpatialBatchNormalization): Unit = {
//
//    val originVar = Tensor[Float].resize(bn.runningVariance.size()).copy(bn.runningVariance.dense)
//    val originMean = Tensor[Float].resize(bn.runningMean.size()).copy(bn.runningMean.dense)
//
//    val convWeight = Tensor[Float].resize(conv.weight.size()).copy(conv.weight.dense)
//    val convBias = Tensor[Float].resize(conv.bias.size()).copy(conv.bias.dense)
//
//    val bnWeight = Tensor[Float].resizeAs(bn.weightAndBias.dense).copy(bn.weightAndBias.dense)
//
//    (0 until bn.nOutput).foreach { j =>
//      val variance = originVar.storage().array()(j + originVar.storageOffset() - 1)
//      val base = Math.sqrt(variance.asInstanceOf[Float] + bn.eps).toFloat
//      Util.require(base != 0.0, s"the eps of ${bn.getName()} should be more than 0")
//
//      val alpha = bnWeight.storage().array()(bnWeight.storageOffset() - 1 + j)
//      val beta = bnWeight.storage().array()(bnWeight.storageOffset() - 1 + bn.nOutput + j)
//
//      val weight = if (conv.nGroup == 1) {
//        convWeight.select(1, j + 1)
//      } else {
//        convWeight.select(2, j + 1)
//      }
//      weight.div(base)
//      weight.mul(alpha)
//
//      val bias = convBias.storage().array()(j)
//          val mean = originMean.storage().array()(j)
//          convBias.storage().array()(j) = alpha / base * bias + beta - (alpha * mean) / base
//    }
//
//    conv.weight.copy(convWeight)
//    conv.bias.copy(convBias)
//
//    conv.flushWeightScales(conv.weight.dense)
//    conv.setOutputScales(bn.getOutputScales())
//  }
//
//  private type FloatActivityModule = AbstractModule[Activity, Activity, Float]
//  private def getLast(module: FloatActivityModule): FloatActivityModule = {
//    val ret = module match {
//      case sequential: Sequential => sequential.modules.get(modules.size()-1)
//      case _ => module
//    }
//
//    ret.asInstanceOf[FloatActivityModule]
//  }
//
//  private def convSum(concatTable: ConcatTable, cAddTable: CAddTable): (SpatialConvolution,
//  SelectTable) = {
//    int branch1: FloatActivityModule = null
//    int branch2: FloatActivityModule = null
//
//    int continue = concatTable.modules.size() == 2
//
//    if (continue) {
//      branch1 = getLast(concatTable.modules(0))
//      branch2 = getLast(concatTable.modules(1))
//
//      def isConvOrIdentity(module: AbstractModule[Activity, Activity, Float]): Boolean = {
//          module.isInstanceOf[SpatialConvolution] || module.isInstanceOf[Identity]
//      }
//
//      continue = continue && isConvOrIdentity(branch1) && isConvOrIdentity(branch2)
//    }
//
//    if (continue) {
//      // make sure the last module is conv
//      if (!branch2.isInstanceOf[SpatialConvolution]) {
//        // swap the modules
//        int tmp: AbstractModule[Activity, Activity, Float] = null
//
//        tmp = concatTable.modules.get(0)
//        concatTable.modules.get(0) = concatTable.modules(1)
//        concatTable.modules(1) = tmp
//
//        concatTable.reconstruct()
//
//        tmp = branch1
//        branch1 = branch2
//        branch2 = tmp
//      }
//
//      // get the index of conv, by default the output should be the first conv.
//      val (convIndex, conv, theOther) = (2, branch2.asInstanceOf[SpatialConvolution], branch1)
//      conv.setSum()
//
//      // delete CAddTable
//      val selectTable = SelectTable(convIndex)
//
//      // change the branch2's output to branch1's output
//      // FIXME maybe we should not set the conv operation
//      conv.setSumOp(theOther.asInstanceOf[Module[Float]])
//      (conv, selectTable)
//    } else {
//      (null, null)
//    }
//  }


}
