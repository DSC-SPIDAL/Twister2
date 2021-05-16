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
package edu.iu.dsc.tws.dl.module;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;

import edu.iu.dsc.tws.api.tset.sets.TSet;
import edu.iu.dsc.tws.dl.data.Activity;
import edu.iu.dsc.tws.dl.data.MiniBatch;
import edu.iu.dsc.tws.dl.data.Sample;
import edu.iu.dsc.tws.dl.data.Table;
import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.graph.Edge;
import edu.iu.dsc.tws.dl.graph.Graph;
import edu.iu.dsc.tws.dl.graph.Node;
import edu.iu.dsc.tws.dl.graph.StaticGraph;
import edu.iu.dsc.tws.dl.optim.OptimMethod;
import edu.iu.dsc.tws.dl.optim.PredictAccuracyMapFunction;
import edu.iu.dsc.tws.dl.optim.PredictClassMapFunction;
import edu.iu.dsc.tws.dl.utils.Util;
import edu.iu.dsc.tws.dl.utils.pair.ModuleNodeIntPair;
import edu.iu.dsc.tws.dl.utils.pair.TensorArrayPair;
import edu.iu.dsc.tws.dl.utils.pair.TensorPair;
import edu.iu.dsc.tws.tset.sets.batch.SourceTSet;

@SuppressWarnings({"MemberName", "HiddenField"})
public abstract class AbstractModule<A extends Activity> extends InferShape
    implements Module, Serializable {

  // ================================= Public APIs =============================================
  /**
   * The cached output. So we don't compute it again when need it
   */
  public A output;

  /**
   * The cached gradient of activities. So we don't compute it again when need it
   */
  public A gradInput;

  protected List<Integer> inputsFormats = null;
  protected List<Integer> outputsFormats = null;
  /**
   * The scale of gradient weight and gradient bias
   * before gradParameters being accumulated.
   */
  protected double scaleW = 1.0;
  protected double scaleB = 1.0;
  protected long forwardTime = 0L;
  protected long backwardTime = 0L;
  /**
   * Module status. It is useful for modules like dropout/batch normalization
   */
  protected boolean train = true;
  protected boolean isFloat = false;
  protected String line = "\n";
  private String namePostfix = Integer.toHexString(java.util.UUID.randomUUID().hashCode());
  private String getNamePostfix = namePostfix;
  /**
   * The name of the module
   */
  private String name = null;
  private int id = 0;
  private double scaleWCache = scaleW;
  private double scaleBCache = scaleB;
  private OptimMethod _optimMethod = null;

  private boolean isMklDnn = false;

  /**
   * Convert the modules to float. If there are any specific changes needed to support float
   * at the module level this method needs to be overridden.
   */
  public void toFloat() {
    this.isFloat = true;
  }

  /**
   * Check if module is float
   *
   * @return true if float
   */
  public boolean isFloat() {
    return isFloat;
  }

  public boolean isMklDnn() {
    return isMklDnn;
  }

  public void setMklDnn(boolean mklDnn) {
    isMklDnn = mklDnn;
  }

  /**
   * set input formats for graph
   *
   * @param formats
   * @return
   */
  public AbstractModule setInputFormats(List<Integer> formats) {
    inputsFormats = formats;
    return this;
  }

  /**
   * set output formats for graph
   *
   * @param formats
   * @return
   */
  public AbstractModule setOutputFormats(List<Integer> formats) {
    outputsFormats = formats;
    return this;
  }

  /**
   * Get the scale of gradientWeight
   */
  public final double getScaleW() {
    return scaleW;
  }

  /**
   * Set the scale of gradientWeight
   *
   * @param w the value of the scale of gradientWeight
   * @return this
   */
  public AbstractModule setScaleW(double w) {
    scaleW = w;
    return this;
  }

  /**
   * Get the scale of gradientBias
   */
  public final double getScaleB() {
    return scaleB;
  }

  /**
   * Set the scale of gradientBias
   *
   * @param b the value of the scale of gradientBias
   * @return this
   */
  public AbstractModule setScaleB(double b) {
    scaleB = b;
    return this;
  }

  /**
   * Clear cached activities to save storage space or network bandwidth. Note that we use
   * Tensor.set to keep some information like tensor share
   * <p>
   * The subclass should override this method if it allocate some extra resource, and call the
   * super.clearState in the override method
   *
   * @return
   */
  public AbstractModule clearState() {
    if (output instanceof Tensor) {
      ((Tensor) output).set();
    }
    if (gradInput instanceof Tensor) {
      ((Tensor) gradInput).set();
    }
    return this;
  }

  /**
   * Whether user set a name to the module before
   *
   * @return
   */
  public final boolean hasName() {
    return name != null;
  }

  /**
   * Get the module name, default name is className@namePostfix
   *
   * @return
   */
  public final String getName() {
    if (this.name == null) {
      return this.getClass().getSimpleName() + namePostfix;
    } else {
      return this.name;
    }
  }

  /**
   * Set the module name
   *
   * @param name
   * @return
   */
  public final AbstractModule setName(String name) {
    this.name = name;
    return this;
  }

  @Override
  public String toString() {
    return getPrintName();
  }

  /**
   * Get the forward/backward cost time for the module or its submodules
   *
   * @return
   */
  public Object[] getTimes() {
    return new Object[]{this, forwardTime, backwardTime};
  }

  /**
   * Get the forward/backward cost time for the module or its submodules
   * and group by module type.
   *
   * @return (module type name, forward time, backward time)
   */
  public final Object[] getTimesGroupByModuleType() {
    throw new UnsupportedOperationException("Operation not supported");
    //    Array[(String, Long, Long)] = {
//      this.getTimes().map(v => (v._1.getClass().getName(), v._2, v._3)).groupBy(_._1)
//          .map(v => (v._1, v._2.reduce((a, b) => (v._1, a._2 + b._2, a._3 + b._3))))
//      .map(v => (v._1, v._2._2, v._2._3))
//      .toArray
//          .sortWith((a, b) => (a._2 + a._3) > (b._2 + b._3))

  }

  /**
   * Reset the forward/backward record time for the module or its submodules
   *
   * @return
   */
  public void resetTimes() {
    forwardTime = 0;
    backwardTime = 0;
  }

  /**
   * freeze the module,
   * i.e. their parameters(weight/bias, if exists) are not changed in training process
   * if names is not empty,
   * set an array of layers that match the given ```names``` to be "freezed",
   *
   * @param names an array of layer names
   * @return current graph model
   */
  public AbstractModule freeze(String[] names) {
    if (names == null || names.length == 0) {
      // in case when freeze is called many times
      if (scaleW != 0) {
        scaleWCache = scaleW;
        scaleW = 0;
      }
      if (scaleB != 0) {
        scaleBCache = scaleB;
        scaleB = 0;
      }
    } else {
      boolean found = false;
      for (String name : names) {
        if (name == this.name) {
          this.freeze(null);
          found = true;
        }
      }
      if (!found) {
        throw new IllegalStateException("cannot match module named $name");
      }
    }
    return this;
  }

  /**
   * "unfreeze" module, i.e. make the module parameters(weight/bias, if exists)
   * to be trained(updated) in training process
   * if names is not empty, unfreeze layers that match given names
   *
   * @param names array of module names to unFreeze
   */
  public AbstractModule unFreeze(String[] names) {
    if (names == null || names.length == 0) {
      scaleW = scaleWCache;
      scaleB = scaleBCache;
    } else {
      boolean found = false;
      for (String name : names) {
        if (name == this.name) {
          this.unFreeze(null);
          found = true;
        }
      }
      if (!found) {
        throw new IllegalStateException("cannot match module named $name");
      }
    }
    return this;
  }

  /**
   * Takes an input object, and computes the corresponding output of the module. After a forward,
   * the output state variable should have been updated to the new value.
   *
   * @param input input data
   * @return output data
   */
  public final A forward(A input) {
    long before = System.nanoTime();
    try {
      updateParameter();
      updateOutput(input);
    } catch (Exception e) {
      throw new IllegalStateException("Layer exception :" + e.getMessage(), e);
//      case l: LayerException =>
//        l.layerMsg = this.toString() + "/" + l.layerMsg
//        throw l
//      case e: Throwable =>
//        throw new LayerException(this.toString(), e);
    }
    forwardTime += System.nanoTime() - before;

    return output;
  }

  /**
   * Performs a back-propagation step through the module, with respect to the given input. In
   * general this method makes the assumption forward(input) has been called before, with the same
   * input. This is necessary for optimization reasons. If you do not respect this rule, backward()
   * will compute incorrect gradients.
   *
   * @param input      input data
   * @param gradOutput gradient of next layer
   * @return gradient corresponding to input data
   */
  public A backward(A input, A gradOutput) {
    long before = System.nanoTime();
    updateGradInput(input, gradOutput);
    accGradParameters(input, gradOutput);
    backwardTime += System.nanoTime() - before;
    asyncGradient();
    return gradInput;
  }

  public void asyncGradient() {
//    if (this.getParameterSynchronizer() != null) {
//      if (this.parameters() != null) {
//        //TODO: param Sync
//        //this.getParameterSynchronizer().put(this.getName());
//      }
//    }
  }

  /**
   * Computes the output using the current parameter set of the class and input. This function
   * returns the result which is stored in the output field.
   *
   * @param input
   * @return
   */
  public abstract A updateOutput(A input);

  /**
   * Computing the gradient of the module with respect to its own input. This is returned in
   * gradInput. Also, the gradInput state variable is updated accordingly.
   *
   * @param input
   * @param gradOutput
   * @return
   */
  public abstract A updateGradInput(A input, A gradOutput);

  /**
   * Computing the gradient of the module with respect to its own parameters. Many modules do not
   * perform this step as they do not have any parameters. The state variable name for the
   * parameters is module dependent. The module is expected to accumulate the gradients with
   * respect to the parameters in some variable.
   *
   * @param input
   * @param gradOutput
   */
  public void accGradParameters(A input, A gradOutput) {

  }

  /**
   * If the module has parameters, this will zero the accumulation of the gradients with respect
   * to these parameters. Otherwise, it does nothing.
   */
  public void zeroGradParameters() {
    if (parameters() != null) {
      for (int i = 0; i < parameters().getValue1().length; i++) {
        Tensor weight = parameters().getValue0()[i];
        Tensor grad = parameters().getValue1()[i];
        grad.resizeAs(weight).zero();
      }
    }
  }

  /**
   * This function returns two arrays. One for the weights and the other the gradients
   * Custom modules should override this function if they have parameters
   *
   * @return (Array of weights, Array of grad)
   */
  public abstract TensorArrayPair parameters();

  /**
   * get the modules of the model.
   * @return
   */
  public List<AbstractModule> getModules() {
    throw new UnsupportedOperationException("not supported");
  }

  /**
   * set the modules during iterations.
   * @param modules
   */
  public void setModules(List<AbstractModule> modules) {
    throw new UnsupportedOperationException("not supported");
  }

  /**
   * Get extra parameter in this module.
   * Extra parameter means the trainable parameters beside weight and bias. Such as runningMean
   * and runningVar in BatchNormalization.
   * <p>
   * The subclass should override this method if it has some parameters besides weight and bias.
   *
   * @return an array of tensor
   */
  public Tensor[] getExtraParameter() {
    return null;
  }

//  /**
//   * Save this module to path in torch7 readable format
//   * @param path
//   * @param overWrite
//   * @return
//   */
//  final def saveTorch(String path, boolean overWrite) {
//    this.clearState()
//    File.saveTorch(this, path, TYPE_MODULE, overWrite)
//    return this;
//  }
//
//  /**
//   * Save this module to path in caffe readable format
//   * @param prototxtPath
//   * @param modelPath
//   * @param useV2
//   * @param overwrite
//   * @return
//   */
//  final def saveCaffe(prototxtPath: String, modelPath: String,
//                      useV2 : Boolean = true, overwrite : Boolean = false) {
//    this.clearState()
//    CaffePersister.persist[T] (prototxtPath, modelPath, this, useV2, overwrite)
//    return this;
//  }
//
//  /**
//   * Save this module to path in tensorflow readable format
//   * @param inputs
//   * @param path
//   * @param byteOrder
//   * @param dataFormat
//   * @return
//   */
//  final def saveTF(
//      inputs : Seq[(String, Seq[Int])],
//  path: String,
//  byteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN,
//  dataFormat: TensorflowDataFormat = TensorflowDataFormat.NHWC)
//
//  {
//    require(this.isInstanceOf[Graph[T]], "only Graph container can be saved as Tensorflow model")
//    this.clearState()
//    val inTrainMode = train
//    if (inTrainMode) {
//      this.evaluate()
//    }
//    TensorflowSaver.saveGraph(this.asInstanceOf[Graph[T]], inputs, path, byteOrder, dataFormat)
//    if (inTrainMode) {
//      this.training()
//    }
//    return this;
//  }

//  /**
//   * Get numeric type of module parameters
//   * @return
//   */
//  final def getNumericType(): TensorDataType = {
//    ev.getType()
//  }

  /**
   * Set extra parameter to this module.
   * Extra parameter means the trainable parameters beside weight and bias. Such as runningMean
   * and runningVar in BatchNormalization.
   *
   * @return this
   */
  public AbstractModule setExtraParameter(Tensor[] extraParam) {
    Tensor[] currentExtraParam = this.getExtraParameter();
    if (extraParam != null && currentExtraParam != null) {
      Util.require(extraParam.length == currentExtraParam.length,
          "state's length doesn't match, excepted:"
              + "${currentExtraParam.length}, but got  ${extraParam.length}");
      int i = 0;
      while (i < extraParam.length) {
        currentExtraParam[i].copy(extraParam[i]);
        i += 1;
      }
      return this;
    } else if (extraParam == null && currentExtraParam == null) {
      return this;
    } else {
      throw new IllegalArgumentException("module's extraParameter is $currentExtraParam"
          + ", while setting param is ${extraParam}");
    }
  }

  /**
   * This function returns a table contains ModuleName, the parameter names and parameter value
   * in this module.
   * <p>
   * The result table is a structure of Table(ModuleName -> Table(ParameterName -> ParameterValue)),
   * and the type is Table[String, Table[String, Tensor[T]]].
   * <p>
   * For example, get the weight of a module named conv1:
   * table[Table]("conv1")[Tensor[T]]("weight").
   * <p>
   * The names of the parameters follow such convention:
   * <p>
   * 1. If there's one parameter, the parameter is named as "weight", the gradient is named as
   * "gradWeight"
   * <p>
   * 2. If there're two parameters, the first parameter is named as "weight", the first gradient is
   * named as "gradWeight"; the second parameter is named as "bias", the seconcd gradient is
   * named as "gradBias"
   * <p>
   * 3. If there're more parameters, the weight is named as "weight" with a seq number as suffix,
   * the gradient is named as "gradient" with a seq number as suffix
   * <p>
   * Custom modules should override this function the default impl if the convention doesn't meet
   * the requirement.
   *
   * @return Table
   */
  public Table getParametersTable() {
    TensorArrayPair params = parameters();
    if (params == null) {
      return null;
    }
    Tensor[] weights = params.getValue0();
    Tensor[] gradients = params.getValue1();
    Util.require(gradients.length == weights.length,
        "weight number is not equal to grad number");

    Table result = new Table();
    if (weights.length == 1) {
      Table temp = new Table();
      temp.put("weight", weights[0]);
      temp.put("gradWeight", gradients[0]);
      result.put(getName(), temp);
    } else if (weights.length == 2) {
      Table temp = new Table();
      temp.put("weight", weights[0]);
      temp.put("bias", weights[1]);
      temp.put("gradWeight", gradients[0]);
      temp.put("gradBias", gradients[1]);
      result.put(getName(), temp);
    } else {
      Table temp = new Table();
      for (int i = 0; i < weights.length; i++) {
        temp.put("weight" + i, weights[i]);
        temp.put("gradient" + i, gradients[i]);
      }
      result.put(getName(), temp);
    }
    return result;
  }

//  /**
//   * model predict images, return imageFrame with predicted tensor,
//   * if you want to call predictImage multiple times,
//   * it is recommended to use Predictor for DistributedImageFrame
//   * or LocalPredictor for LocalImageFrame
//   * @param imageFrame imageFrame that contains images
//   * @param outputLayer if outputLayer is not null, the output of layer that matches
//   *                      outputLayer will be used as predicted output
//   * @param shareBuffer whether to share same memory for each batch predict results
//   * @param batchPerPartition batch size per partition, default is 4
//   * @param predictKey key to store predicted result
//   * @param featurePaddingParam featurePaddingParam if the inputs have variant size
//   * @return
//   */
//  public final def predictImage(imageFrame: ImageFrame,
//                         outputLayer: String = null,
//                         shareBuffer: Boolean = false,
//                         batchPerPartition: Int = 4,
//                         predictKey: String = ImageFeature.predict,
//                         featurePaddingParam: Option[PaddingParam[T]] = None): ImageFrame = {
//    imageFrame match {
//      case distributedImageFrame: DistributedImageFrame =>
//        Predictor(this, featurePaddingParam, batchPerPartition)
//            .predictImage(distributedImageFrame, outputLayer, shareBuffer, predictKey)
//      case localImageFrame: LocalImageFrame =>
//        val predictor = LocalPredictor(this, featurePaddingParam, batchPerPartition)
//        val imageFrame = predictor.predictImage(localImageFrame, outputLayer, shareBuffer,
//            predictKey)
//        predictor.shutdown()
//        imageFrame
//    }
//  }

  /**
   * Set the module to training mode
   *
   * @return
   */
  public AbstractModule training() {
    train = true;
    return this;
  }

  /**
   * Set the module to evaluate mode
   *
   * @return
   */
  public AbstractModule evaluate() {
    train = false;
    return this;
  }

  /**
   * Check if the model is in training mode
   *
   * @return
   */
  public final boolean isTraining() {
    return this.train;
  }

  /**
   * Reset module parameters, which is re-initialize the parameter with given initMethod
   */
  public abstract void reset();

  /**
   * Set the line separator when print the module
   *
   * @param line
   * @return
   */
  public final AbstractModule setLine(String line) {
    this.line = line;
    return this;
  }

  /**
   * Clone the model
   *
   * @return
   */
  public final void cloneModule() {
    throw new UnsupportedOperationException("Operation not supported");
    //SerializationUtils.clone(this);
  }

  /**
   * Clone the module, deep or shallow copy
   *
   * @param deepCopy
   * @return
   */
  public final AbstractModule clone(boolean deepCopy) {
    throw new UnsupportedOperationException("Operation not supported");

//    val moduleData = ModuleData[T](this, Seq[String](), Seq[String]())
//    val storages = new mutable.HashMap[Int, Any]()
//    val context = SerializeContext(moduleData, storages, ProtoStorageType, false)
//    val serializedModule = ModuleSerializer.serialize[T](context).bigDLModule
//    ModulePersister.setTensorStorage(serializedModule, storages)
//
//    storages.clear()
//
//    val deserializeContext = DeserializeContext(serializedModule.build,
//        storages, ProtoStorageType, false)
//    ModuleLoader.initTensorStorage[T](deserializeContext)
//        val copy = ModuleSerializer.load[T](deserializeContext).module
//        .asInstanceOf[AbstractModule[A
//      , A, T]]
//    setWeightAndBias(copy, deepCopy)
//    copy
  }

  @Override
  public int hashCode() {
    List state = new ArrayList();
    state.add(this.output);
    state.add(this.gradInput);
    state.add(this.getClass());
    state.add(this.name);
    return state.stream().mapToInt(o ->
        (o == null) ? 0 : o.hashCode()).reduce(0, (a, b) -> 31 * a + b);
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof AbstractModule
        && output == ((AbstractModule) other).output
        && gradInput == ((AbstractModule) other).gradInput
        && name == ((AbstractModule) other).getName();
  }

  /**
   * Save this module to path with protobuf format
   *
   * @param path       path to save module, local file system, HDFS and Amazon S3 is supported.
   *                   HDFS path should be like "hdfs://[host]:[port]/xxx"
   *                   Amazon S3 path should be like "s3a://bucket/xxx"
   * @param weightPath where to store weight default null
   * @param overWrite  if overwrite default false
   * @return self
   */
  public final AbstractModule saveModule(String path, String weightPath,
                                         boolean overWrite) {
    this.clearState();
    //ModulePersister.saveToFile(path, weightPath, this, overWrite);
    return this;
  }

  /**
   * Save this module definition to path.
   *
   * @param path      path to save module, local file system, HDFS and Amazon S3 is supported.
   *                  HDFS path should be like "hdfs://[host]:[port]/xxx"
   *                  Amazon S3 path should be like "s3a://bucket/xxx"
   * @param overWrite if overwrite
   * @return self
   */
  public final AbstractModule saveDefinition(String path, boolean overWrite) {
    this.clearState();
    //ModulePersister.saveModelDefinitionToFile(path, this, overWrite)
    return this;
  }

  /**
   * module predict, return the probability distribution
   *
   * @param dataset     dataset for prediction
   * @param batchSize   total batchSize for all partitions.
   *                    if -1, default is 4 * partitionNumber of datatset
   * @param shareBuffer whether to share same memory for each batch predict results
   */
  public final TSet<Activity> predict(TSet<Sample> dataset, int batchSize, boolean shareBuffer) {
    //Predictor(this).predict(dataset, batchSize, shareBuffer)
    throw new UnsupportedOperationException("Opretion not supported for Tset Yet");
  }

//  /**
//   * use ValidationMethod to evaluate module on the given rdd dataset
//   * @param dataset dataset for test
//   * @param vMethods validation methods
//   * @param batchSize total batchsize of all partitions,
//   *                  optional param and default 4 * partitionNum of dataset
//   * @return
//   */
//  public final def evaluate(TSet<Sample> dataset, vMethods: Array[_ <:ValidationMethod[T]],
//      batchSize: Option[Int] = None
//  ): Array[(ValidationResult, ValidationMethod[T])] = {
//    Evaluator(this).test(dataset, vMethods.map(v => v), batchSize)
//  }
//
//
//  /**
//   * use ValidationMethod to evaluate module on the given rdd dataset
//   * @param dataset
//   * @param vMethods
//   * @return
//   */
//  public final def evaluate(
//      dataset: RDD[MiniBatch[T]],
//      vMethods: Array[_ <:ValidationMethod[T]]
//  ): Array[(ValidationResult, ValidationMethod[T])] = {
//    Evaluator(this).testMiniBatch(dataset, vMethods.map(v => v))
//  }

//  /**
//   * use ValidationMethod to evaluate module on the given ImageFrame
//   *  @param imageFrame ImageFrame for valudation
//   *  @param vMethods validation methods
//   *  @param batchSize total batch size of all partitions
//   *  @return
//   */
//
//  public final def evaluateImage(imageFrame: ImageFrame,
//                          vMethods: Array[_ <:ValidationMethod[T]],
//                          batchSize: Option[Int] = None
//  ): Array[(ValidationResult, ValidationMethod[T])] = {
//    require(imageFrame.isDistributed(), "ImageFrame must be distributed")
//    val rdd = imageFrame.toDistributed().rdd.map(imageFeature => {
//    if (imageFeature.isValid) {
//      require(imageFeature.contains(ImageFeature.sample), "ImageFeature must have sample")
//      imageFeature[Sample[T]](ImageFeature.sample)
//    } else {
//      null
//    }
//    }).filter(_ != null)
//    evaluate(rdd, vMethods, batchSize)
//  }
//
//  /**
//   * use ValidationMethod to evaluate module on the given local dataset
//   * @param dataSet
//   * @param vMethods
//   * @return
//   */
//  public final def evaluate(
//      dataSet: LocalDataSet[MiniBatch[T]],
//      vMethods: Array[_ <:ValidationMethod[T]]
//  ): Array[(ValidationResult, ValidationMethod[T])] = {
//    Validator(this, dataSet).test(vMethods.map(v => v))
//  }

//  /**
//   * Quantize this module, which reduces the precision of the parameter. Get a higher speed with a
//   * little accuracy cost.
//   * @return
//   */
//  final def quantize(): Module[T] = {
//    ConversionUtils.convert[T](this, true)
//  }

  // ================================= Internal APIs ===========================================

  /**
   * module predict, return the predict label
   *
   * @param dataset   dataset for prediction
   * @param batchSize total batchSize for all partitions.
   */

  public final int[] predictClass(SourceTSet<MiniBatch> dataset, int batchSize, int dataSize) {
    int[] results = new int[dataSize];
    int index = 0;
    List dataList = dataset.direct().<int[]>map(new PredictClassMapFunction(this))
        .gather().cache().getData();
    for (Object data : dataList) {
      int[] temp = (int[]) data;
      for (int i = 0; i < temp.length; i++) {
        results[index] = temp[i];
        index++;
      }
    }
    return results;
  }

  public double predictAccuracy(SourceTSet<MiniBatch> dataset, int batchSize, int testDataSize) {
    List dataList = dataset.direct().<int[]>map(new PredictAccuracyMapFunction(this))
        .gather().cache().getData();
    if (dataset.getTSetEnv().getWorkerID() == 0) {
      int errorCount = 0;
      int pointCount = 0;
      for (Object data : dataList) {
        int[] temp = (int[]) data;
        pointCount += temp[0];
        errorCount += temp[1];
      }
      Util.require(testDataSize == pointCount, "The point count in the dataset and "
          + "given testDataSize do not match " + testDataSize + " : " + pointCount);
      return 1.0 - ((double) errorCount / testDataSize);
    } else {
      return 0;
    }

  }

  /**
   * Get weight and bias for the module
   *
   * @return array of weights and bias
   */
  public final Tensor[] getWeightsBias() {
    if (parameters() != null) {
      return parameters().getValue0();
    } else {
      return null;
    }
  }

  /**
   * Set weight and bias for the module
   *
   * @param newWeights array of weights and bias
   * @return
   */
  public final AbstractModule setWeightsBias(Tensor[] newWeights) {
    Util.require(parameters() != null, "this layer does not have weight/bias");
    Util.require(parameters().getValue0().length == newWeights.length,
        "the number of input weight/bias is not consistant with "
            + "number of weight/bias of this layer, "
            + "number of input ${parameters()._1.length},"
            + " number of output ${newWeights.length}");

    Tensor[] weights = parameters().getValue0();


    for (int i = 0; i < newWeights.length; i++) {
      // TODO: enable this checking as we don't respect shape right now.
      //      require(weights(i).size().deep == newWeights(i).size().deep,
      //        s"Mismatch shape, ${weights(i).size().mkString(",")}" +
      //          s" vs ${newWeights(i).size().mkString(",")} ")
      weights[i].copy(newWeights[i]);
    }
    return this;
  }

  /**
   * save weights and bias to file
   *
   * @param path      file to save
   * @param overWrite whether to overwrite or not
   */
  public final void saveWeights(String path, boolean overWrite) {
    throw new UnsupportedOperationException("Operation not supported");
//    val parameterTable = getParametersTable()
//    val weightsBiasTable = T()
//    parameterTable.foreach {
//      case (name: String, params: Table) =>
//        val wb = T()
//        if (params.contains("weight")) {
//          wb("weight") = params("weight")
//        }
//        if (params.contains("bias")) {
//          wb("bias") = params("bias")
//        }
//        weightsBiasTable(name) = wb
//      case _ => throw new UnsupportedOperationException("invalid parameter table")
//    }
//    weightsBiasTable.save(path, overWrite)
  }

  /**
   * load pretrained weights and bias to current module
   *
   * @param weightPath file to store weights and bias
   * @param matchAll   whether to match all layers' weights and bias,
   *                   if not, only load existing pretrained weights and bias
   * @return current module
   */
  public final AbstractModule loadWeights(String weightPath, boolean matchAll) {
    throw new UnsupportedOperationException("Operation not supported");
//    val srcParameter = File.load[Table] (weightPath)
//        val targetParameter = getParametersTable()
//    copyWeights(targetParameter, srcParameter, matchAll)
//    return this;
  }

//  private[nn] final def allocateAs(dest: Activity): Activity = dest match {
//    case tensor: Tensor[T] => Tensor[T]()
//    case table: Table => T()
//    case _ => throw new IllegalArgumentException("Activity only support tensor and table now")
//  }

  /**
   * copy weights from another model, mapping by layer name
   *
   * @param srcModel model to copy from
   * @param matchAll whether to match all layers' weights and bias,
   * @return current module
   */
  public final AbstractModule loadModelWeights(Module srcModel, boolean matchAll) {
    throw new UnsupportedOperationException("Operation not supported");
    //    val srcParameter = srcModel.getParametersTable()
//    val targetParameter = getParametersTable()
//    copyWeights(targetParameter, srcParameter, matchAll)
//    return this;
  }

  protected Node<AbstractModule> processInputs(List<Node<AbstractModule>> nodes) {
    Node<AbstractModule> curNode = new Node<AbstractModule>(this);
    nodes.stream().forEach(node -> node.add(curNode, new Edge()));
    return curNode;
  }

  protected Node<AbstractModule> processInputs(ModuleNodeIntPair first,
                                               ModuleNodeIntPair... nodesWithIndex) {
    Node<AbstractModule> curNode = new Node<AbstractModule>(this);
    first.getValue0().add(curNode, new Edge(first.getValue1()));
    Arrays.stream(nodesWithIndex).sequential()
        .forEach(nodeWithIndex
            -> nodeWithIndex.getValue0().add(curNode, new Edge(nodeWithIndex.getValue1())));
    return curNode;
  }

  /**
   * Build graph: some other modules point to current module
   *
   * @param nodes upstream module nodes
   * @return node containing current module
   */
  public Node<AbstractModule> inputs(Node<AbstractModule>... nodes) {
    validateInput(Arrays.stream(nodes)
        .map(moduleNode -> moduleNode.getElement()).collect(Collectors.toList()));
    return processInputs(Arrays.asList(nodes));
  }

  /**
   * Build graph: some other modules point to current module
   *
   * @param nodes upstream module nodes in an array
   * @return node containing current module
   */
  public Node<AbstractModule> inputsArr(Node<AbstractModule>[] nodes) {
    validateInput(Arrays.stream(nodes)
        .map(moduleNode -> moduleNode.getElement()).collect(Collectors.toList()));
    return processInputs(Arrays.asList(nodes));
  }

  /**
   * Build graph: some other modules point to current module
   *
   * @param first          distinguish from another inputs when input parameter list is empty
   * @param nodesWithIndex upstream module nodes and the output tensor index. The start index is 1.
   * @return node containing current module
   */
  public Node<AbstractModule> inputs(ModuleNodeIntPair first, ModuleNodeIntPair... nodesWithIndex) {
    validateInput(Arrays.asList(first.getValue0().getElement()));
    validateInput(Arrays.stream(nodesWithIndex)
        .map(pair -> pair.getValue0().getElement()).collect(Collectors.toList()));
    return processInputs(first, nodesWithIndex);
  }

  /**
   * Generate graph module with start nodes
   *
   * @param startNodes
   * @return
   */
  public Graph toGraph(Node<AbstractModule>... startNodes) {
    Node<AbstractModule>[] starts;

    if (startNodes == null || startNodes.length == 0) {
      starts = new Node[]{new Node<AbstractModule>(new Input())};
    } else {
      starts = startNodes;
    }

    Node<AbstractModule>[] endNodes = this.getEndNodes(starts);
    Graph graph = new StaticGraph(Arrays.asList(starts), Arrays.asList(endNodes));
    if (graph instanceof StaticGraph) {
      // Merge nested graphs inside to make the whole graph non-nested
      graph = ((StaticGraph) graph).toSingleGraph();
    }
    if (inputsFormats != null) {
      graph.setInputFormats(inputsFormats);
    }
    if (outputsFormats != null) {
      graph.setOutputFormats(outputsFormats);
    }
    return graph;
  }

  /**
   * Find a module with given name. If there is no module with given name, it will return None. If
   * there are multiple modules with the given name, an exception will be thrown.
   *
   * @param name
   * @return
   */
  public AbstractModule apply(String name) {
    if (this.getName() == name) {
      return this;
    } else {
      return null;
    }
  }

  private void setNamePostfix(String namePostfix) {
    this.namePostfix = namePostfix;
  }

  private int getId() {
    return this.id;
  }

  private void setId(int id) {
    this.id = id;
  }

  protected final String getPrintName() {
    String postfix = (name == null) ? namePostfix : name;
    return this.getClass().getSimpleName() + "[" + postfix + "]";

  }


  //private val engineType: EngineType = Engine.getEngineType()

  /**
   * This function returns two tensors. One for the flattened trainable parameters flatParameters
   * and another for the gradients of the energy wrt to the trainable parameters flatGradParameters.
   * <p>
   * Custom modules should not override this function. They should instead override parameters(...)
   * which is, in turn, called by the present function.
   * <p>
   * This function will go over all the weights and gradWeights and make them view into a single
   * tensor (one for weights and one for gradWeights).
   *
   * @return
   */
  public final TensorPair getParameters() {
    TensorArrayPair weightAndGradParameters = this.parameters();

    // maybe null if not weights in this module.
    Util.require(weightAndGradParameters.getValue0() != null
            && weightAndGradParameters.getValue0().length > 0,
        "model ${this.getName()} doesn't have any trainable parameters.");

    // If some gradParameters are not allocated storage, allocate it
    Util.require(
        weightAndGradParameters.getValue0().length == weightAndGradParameters.getValue1().length,
        "weights and gradient number are not match");

    for (int i = 0; i < weightAndGradParameters.getValue0().length; i++) {
      weightAndGradParameters.getValue1()[i].resizeAs(weightAndGradParameters.getValue0()[i]);
    }
    return new TensorPair(ModuleUtil.flatten(weightAndGradParameters.getValue0()),
        ModuleUtil.flatten(weightAndGradParameters.getValue1()));
  }

  /**
   * get execution engine type
   */
//  private def checkEngineType() {
//    if (engineType != Engine.getEngineType()) {
//      throw new Error("Module's EngineType doesn't march global EngineType");
//    }
//    return this;
//  }
  private void setWeightAndBias(AbstractModule copy, boolean deepCopy) {
    Table parameterTable = this.getParametersTable();
    Table copiedModuleParamTable = copy.getParametersTable();
    if (parameterTable != null) {
      Util.require(copiedModuleParamTable != null, "cloned module should have params");
      parameterTable.forEach((nameKey, params) -> {
        if (nameKey instanceof String && params instanceof Table) {
          Util.require(copiedModuleParamTable.get(nameKey) != null,
              "cloned module should have for $name");
          setLayerWeightAndBias((Table) params,
              copiedModuleParamTable.<Table>get(nameKey), deepCopy);
        } else {
          throw new UnsupportedOperationException("unsupported $name and $params");
        }
      });
    }
  }

  private void setLayerWeightAndBias(Table params, Table copyParams, boolean deepCopy) {
    params.forEach((param1, param2) -> copyParam(params, copyParams, deepCopy, (String) param1));
  }

  private void copyParam(Table params, Table copyParams,
                         boolean deepCopy, String paraName) {
    if (params.contains(paraName)) {
      // this is for quantization tensors where the weight might be an array
      if (params.get(paraName) instanceof Tensor[]) {
        Tensor[] copies = copyParams.<Tensor[]>get(paraName);
        Tensor[] origins = params.<Tensor[]>get(paraName);
        int i = 0;
        while (i < copies.length) {
          copyTensor(origins[i], copies[i], deepCopy);
          i += 1;
        }
      } else {
        // For normal layers, their params are just tensors
        copyTensor(params.get(paraName), copyParams.get(paraName), deepCopy);
      }
    }
  }

  private void copyTensor(Tensor t1, Tensor t2, boolean deepCopy) {
//    if (t2.isInstanceOf[QuantizedTensor[_]]) {
//      t2.asInstanceOf[QuantizedTensor[_]].release()
//    }
    if (deepCopy) {
      t2.copy(t1);
    } else {
      t2.set(t1);
    }
  }

  private void copyWeights(Table target, Table src, boolean matchAll) {
    target.forEach((nameKey, targetParams) -> {
      if (nameKey instanceof String && targetParams instanceof Table) {
        if (src.contains(nameKey)) {
          Table srcParams = src.<Table>get(nameKey);
          if (srcParams.contains("weight")) {
            Tensor w = srcParams.<Tensor>get("weight");
            ((Table) targetParams).<Tensor>get("weight").resizeAs(w).copy(w);
          }
          if (srcParams.contains("bias")) {
            Tensor b = srcParams.<Tensor>get("bias");
            ((Table) targetParams).<Tensor>get("bias").resizeAs(b).copy(b);
          }
        } else {
          if (matchAll) {
            throw new IllegalStateException("module"
                + nameKey + " cannot find corresponding weight bias");
          }
        }
      } else {
        throw new UnsupportedOperationException("unsupported $name and $targetParams");
      }
    });
  }

//  /**
//   * Return classTag numerics for module serialization. If your module contains multiple classtag
//   * in the constructor, you should override this method
//   * @return
//   */
//  private def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
//    (Array(scala.reflect.classTag[T]), Array(ev))
//  }

  /**
   * Generate end nodes of current module with start nodes
   *
   * @param startNodes current start nodes
   * @return current end nodes
   */
  public Node<AbstractModule>[] getEndNodes(Node<AbstractModule>[] startNodes) {
    Node<AbstractModule>[] endNodes = new Node[]{this.processInputs(Arrays.asList(startNodes))};
    return endNodes;
  }

  /**
   * Check if some module is duplicated in the model. For a layer it cannot be duplicated.
   * Container should override this method
   */
  protected void checkDuplicate(HashSet<Integer> record) {
    String errMsg = "Some module is duplicate in the current model: ";
    int curId = System.identityHashCode(this);
    Util.require(this.skipDuplicateCheck() || !record.contains(curId), errMsg + this.getName());
    record.add(curId);
  }

  /**
   * Sometimes, some layer need skip the duplicate check process, e.g. Keras-like input layer
   *
   * @return
   */
  protected boolean skipDuplicateCheck() {
    return false;
  }

//  /**
//   * parameter synchronizer for gradient synchronization
//   */
//  private var _parameterSynchronizer: DistriParameterSynchronizer[T] = null
//
//  /**
//   * set parameter synchronizer
//   * @param parameterSynchronizer parameter synchronizer
//   */
//  private def setParameterSynchronizer(parameterSynchronizer:
//      DistriParameterSynchronizer[T]): Unit = {
//    _parameterSynchronizer = parameterSynchronizer
//  }
//
//
//  /**
//   * get parameter synchronizer
//   * @return parameter synchronizer
//   */
//  private def getParameterSynchronizer():
//  DistriParameterSynchronizer[T] = _parameterSynchronizer

  /**
   * if the model contains native resources such as aligned memory, we should release it by manual.
   * JVM GC can't release them reliably.
   */
  public void release() {

  }

  /**
   * get optim method for layer
   */

  private OptimMethod getOptimMethod() {
    return this._optimMethod;
  }

  /**
   * set optim method
   */

  private void setOptimMethod(OptimMethod optimMethod) {
    _optimMethod = optimMethod;
  }

  protected void updateParameter() {
//    if (this.getParameterSynchronizer() != null && this.isTraining()) {
//      throw new UnsupportedOperationException("Opretion not supported for Tset Yet");

    //      if (this.parameters() != null) {
//        long before = System.nanoTime();
//        val (weights, grads) = this.getParameterSynchronizer().get(this.getName());
//        val syncEndTime = System.nanoTime()
//        if (grads != null) {
//          val optimMethod = this.getOptimMethod
//          require(optimMethod != null, s"optim method for ${this.getName} cannot be null")
//          optimMethod.optimize(_ => (ev.fromType(0.0f), grads),
//          weights)
//          this.zeroGradParameters
//        }
//      }
//    }
  }
}

