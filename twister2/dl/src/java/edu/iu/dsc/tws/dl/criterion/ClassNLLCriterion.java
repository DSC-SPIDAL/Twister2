//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless Util.required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
package edu.iu.dsc.tws.dl.criterion;

import edu.iu.dsc.tws.dl.data.Tensor;
import edu.iu.dsc.tws.dl.data.TensorNumeric;
import edu.iu.dsc.tws.dl.data.tensor.DenseTensor;
import edu.iu.dsc.tws.dl.utils.ErrorConstants;
import edu.iu.dsc.tws.dl.utils.Util;

/**
 * The negative log likelihood criterion. It is useful to train a classification problem with n
 * classes. If provided, the optional argument weights should be a 1D Tensor assigning weight to
 * each of the classes. This is particularly useful when you have an unbalanced training set.
 * <p>
 * The input given through a forward() is expected to contain log-probabilities/probabilities of
 * each class: input has to be a 1D Tensor of size n. Obtaining log-probabilities/probabilities
 * in a neural network is easily achieved by adding a LogSoftMax/SoftMax layer in the last layer
 * of your neural network. You may use CrossEntropyCriterion instead, if you prefer not to add
 * an extra layer to your network. This criterion expects a class index (1 to the number of class)
 * as target when calling forward(input, target) and backward(input, target).
 * <p>
 * In the log-probabilities case,
 * The loss can be described as:
 * loss(x, class) = -x[class]
 * or in the case of the weights argument it is specified as follows:
 * loss(x, class) = -weights[class] * x[class]
 * <p>
 * Due to the behaviour of the backend code, it is necessary to set sizeAverage to false when
 * calculating losses in non-batch mode.
 * <p>
 * Note that if the target is `paddingValue`, the training process will skip this sample.
 * In other words, the forward process will return zero output and the backward process
 * will also return zero `gradInput`.
 * <p>
 * By default, the losses are averaged over observations for each minibatch. However, if the field
 * sizeAverage is set to false, the losses are instead summed for each minibatch.
 * <p>
 * In particular, when weights=None, size_average=True and logProbAsInput=False, this is same as
 * `sparse_categorical_crossentropy` loss in keras.
 *
 * @param weights        weights of each element of the input
 * @param sizeAverage    size average of batch
 * @param logProbAsInput indicating whether to accept log-probabilities or probabilities as input.
 *                       True means accepting log-probabilities as input.
 */

@SuppressWarnings("MemberName")
public class ClassNLLCriterion extends TensorCriterion {

  private DenseTensor weights;
  private boolean sizeAverage;
  private boolean logProbAsInput;
  private int paddingValue;

  private double total_weight = 0.0;


  //private transient results: Array[Future[(T, T)]] = null
  //private transient var resultsBackward: Array[Future[_]] = null

  private double epsilon = 1e-8;

  private double oneMinusEpsilon = TensorNumeric.minus(1, epsilon);

  public ClassNLLCriterion(DenseTensor weights, boolean sizeAverage,
                           boolean logProbAsInput, int paddingValue) {
    this.weights = weights;
    this.sizeAverage = sizeAverage;
    this.logProbAsInput = logProbAsInput;
    this.paddingValue = paddingValue;

    if (weights != null) {
      Util.require(weights.dim() == 1,
          "weights input should be 1-D Tensor weights dim(${weights.dim()})");
    }

    if (sizeAverage) {
      sizeAverageStatus = SizeAverageStatus.TRUE;
    } else {
      sizeAverageStatus = SizeAverageStatus.FALSE;

    }

  }

  public ClassNLLCriterion(DenseTensor weights, boolean sizeAverage) {
    this(weights, sizeAverage, true, -1);
  }

  public ClassNLLCriterion() {
    this(null, true, true, -1);
  }

  @Override
  public double updateOutput(Tensor input, Tensor target) {
    Util.require(input.dim() == 1 || input.dim() == 2,
        "ClassNLLCriterion: "
            + ErrorConstants.constrainInputAsVectorOrBatch
            + "input dim(${input.dim()})");
    int nClasses = input.size(input.dim());
    if (input.dim() == 1) {
      Util.require(input.dim() == target.dim(),
          "ClassNLLCriterion: " + ErrorConstants.constrainInputDimSameAsTarget
              + " Input dimension is: ${ input.dim() } , target dimension is: ${ target.dim() }");
      int curTarget = (int) target.valueAt(1);
      if (curTarget < 1 && curTarget > nClasses || curTarget == paddingValue) {
        throw new IllegalStateException("curTarget ${curTarget} is out of range,"
            + " should be 1 to ${nClasses}");
      }

      if (weights != null) {
        total_weight = weights.apply(new int[]{curTarget});
      } else {
        total_weight = 1;
      }

      if (curTarget == paddingValue) {
        output = 0;
      } else {
        if (!logProbAsInput) {
          double clipped = TensorNumeric.clip(input.valueAt(curTarget), epsilon, oneMinusEpsilon);
          TensorNumeric.times(TensorNumeric.negative(TensorNumeric.log(clipped)), total_weight);
        } else {
          TensorNumeric.times(TensorNumeric.negative(input.valueAt(curTarget)), total_weight);
        }
      }
    } else if (input.dim() == 2) {
      int batchSize = input.size(1);
      int[] targetSize = target.size();
      target.squeeze();
      Util.require(target.dim() == 1,
          "ClassNLLCriterion: illegal target! Target should be 1D tensor after squeeze,"
              + "but target's size is: ${ target.size() }, please check your data.");

      total_weight = 0;
      output = 0;


      for (int i = 1; i <= batchSize; i++) {

        int curTarget = (int) target.valueAt(i);
        if (curTarget < 1 && curTarget > nClasses || curTarget == paddingValue) {
          throw new IllegalStateException("curTarget " + curTarget
              + "is out of range 1 to ${nClasses}");
        }
        if (curTarget == paddingValue) {
          output = TensorNumeric.minus(output, 0);
          total_weight = TensorNumeric.plus(total_weight, 0);
        } else {
          double curWeight;
          if (weights != null) {
            curWeight = weights.valueAt(curTarget);
          } else {
            curWeight = 1;
          }

          if (!logProbAsInput) {
            double clipped = TensorNumeric.clip(input.valueAt(i, curTarget),
                epsilon, oneMinusEpsilon);
            output = TensorNumeric.minus(output,
                TensorNumeric.times(TensorNumeric.log(clipped), curWeight));
            total_weight = TensorNumeric.plus(total_weight, curWeight);
          } else {
            output = TensorNumeric.minus(output,
                TensorNumeric.times(input.valueAt(i, curTarget), curWeight));
            total_weight = TensorNumeric.plus(total_weight, curWeight);
          }
        }
      }

      if (total_weight == 0) {
        total_weight = 1;
      }
      target.resize(targetSize);
    }
    if (sizeAverage && total_weight != 0) {
      output = TensorNumeric.divide(output, total_weight);
    }
    return output;
  }

  @Override
  public float updateOutputf(Tensor input, Tensor target) {
    Util.require(input.dim() == 1 || input.dim() == 2,
        "ClassNLLCriterion: "
            + ErrorConstants.constrainInputAsVectorOrBatch
            + "input dim(${input.dim()})");
    int nClasses = input.size(input.dim());
    if (input.dim() == 1) {
      Util.require(input.dim() == target.dim(),
          "ClassNLLCriterion: " + ErrorConstants.constrainInputDimSameAsTarget
              + " Input dimension is: ${ input.dim() } , target dimension is: ${ target.dim() }");
      int curTarget = (int) target.valueAtf(1);
      if (curTarget < 1 && curTarget > nClasses || curTarget == paddingValue) {
        throw new IllegalStateException("curTarget ${curTarget} is out of range,"
            + " should be 1 to ${nClasses}");
      }

      if (weights != null) {
        total_weight = weights.applyf(new int[]{curTarget});
      } else {
        total_weight = 1.0;
      }

      if (curTarget == paddingValue) {
        outputf = 0.0f;
      } else {
        if (!logProbAsInput) {
          float clipped = TensorNumeric.clip(input.valueAtf(curTarget), (float) epsilon,
              (float) oneMinusEpsilon);
          TensorNumeric.times(TensorNumeric.negative(TensorNumeric.log(clipped)),
              (float) total_weight);
        } else {
          TensorNumeric.times(TensorNumeric.negative(input.valueAtf(curTarget)),
              (float) total_weight);
        }
      }
    } else if (input.dim() == 2) {
      int batchSize = input.size(1);
      int[] targetSize = target.size();
      target.squeeze();
      Util.require(target.dim() == 1,
          "ClassNLLCriterion: illegal target! Target should be 1D tensor after squeeze,"
              + "but target's size is: ${ target.size() }, please check your data.");

      total_weight = 0.0;
      outputf = 0.0f;


      for (int i = 1; i <= batchSize; i++) {

        int curTarget = (int) target.valueAtf(i);
        if (curTarget < 1 && curTarget > nClasses || curTarget == paddingValue) {
          throw new IllegalStateException("curTarget " + curTarget
              + "is out of range 1 to ${nClasses}");
        }
        if (curTarget == paddingValue) {
          outputf = TensorNumeric.minus(outputf, 0.0f);
          total_weight = TensorNumeric.plus((float) total_weight, 0.0f);
        } else {
          float curWeight;
          if (weights != null) {
            curWeight = weights.valueAtf(curTarget);
          } else {
            curWeight = 1;
          }

          if (!logProbAsInput) {
            float clipped = TensorNumeric.clip(input.valueAtf(i, curTarget),
                (float) epsilon, (float) oneMinusEpsilon);
            outputf = TensorNumeric.minus(outputf,
                TensorNumeric.times(TensorNumeric.log(clipped), curWeight));
            total_weight = TensorNumeric.plus(total_weight, curWeight);
          } else {
            outputf = TensorNumeric.minus(outputf,
                TensorNumeric.times(input.valueAtf(i, curTarget), curWeight));
            total_weight = TensorNumeric.plus(total_weight, curWeight);
          }
        }
      }

      if (total_weight == 0) {
        total_weight = 1;
      }
      target.resize(targetSize);
    }
    if (sizeAverage && total_weight != 0) {
      outputf = TensorNumeric.divide(outputf, (float) total_weight);
    }
    return outputf;
  }

  @Override
  public Tensor updateGradInput(Tensor input, Tensor target) {
    Util.require(input.dim() == 1 || input.dim() == 2,
        "ClassNLLCriterion: "
            + ErrorConstants.constrainInputAsVectorOrBatch + "input dim ${input.dim()}");

    if (total_weight <= 0.0) {
      throw new IllegalStateException("total weight must larger than 0");
    }

    gradInput.resizeAs(input);
    gradInput.zero();
    if (this.isFloat) {
      if (input.dim() == 1) {
        Util.require(input.dim() == target.dim(),
            "ClassNLLCriterion: " + ErrorConstants.constrainInputDimSameAsTarget
                + " Input dimension is: ${ input.dim() } , target dimension is: ${ target.dim() }");
        int curTarget = (int) target.valueAtf(1);
        if (curTarget == paddingValue) {
          return gradInput;
        }

        float temp = -1;
        if (weights != null) {
          temp = TensorNumeric.times(-1, weights.valueAtf(curTarget));
        }
        gradInput.setValue(curTarget, temp);

        if (sizeAverage) {
          gradInput.setValue(curTarget, TensorNumeric.divide(gradInput.valueAtf(curTarget),
              (float) total_weight));
        }
        if (!logProbAsInput) {
          float clipped = TensorNumeric.clip(input.valueAtf(curTarget), (float) epsilon,
              (float) oneMinusEpsilon);
          gradInput.setValue(curTarget,
              TensorNumeric.times(gradInput.valueAtf(curTarget), TensorNumeric.inv(clipped)));
        }
      } else if (input.dim() == 2) {
        int batchSize = input.size(1);
        int[] targetSize = target.size();
        target.squeeze();

        for (int i = 1; i <= batchSize; i++) {
          int curTarget = (int) target.valueAtf(i);
          if (curTarget != paddingValue) {
            float temp = -1;
            if (weights != null) {
              temp = TensorNumeric.times(-1, weights.valueAtf(curTarget));
            }

            gradInput.setValue(i, curTarget, temp);
            if (sizeAverage) {
              gradInput.setValue(i, curTarget, TensorNumeric.divide(gradInput.valueAtf(i,
                  curTarget), (float) total_weight));
            }
            if (!logProbAsInput) {
              float clipped = TensorNumeric.clip(input.valueAtf(i, curTarget),
                  (float) epsilon, (float) oneMinusEpsilon);
              gradInput.setValue(i, curTarget,
                  TensorNumeric.times(gradInput.valueAtf(i, curTarget),
                      TensorNumeric.inv(clipped)));
            }
          }
        }
        target.resize(targetSize);
      }
    } else {
      if (input.dim() == 1) {
        Util.require(input.dim() == target.dim(),
            "ClassNLLCriterion: " + ErrorConstants.constrainInputDimSameAsTarget
                + " Input dimension is: ${ input.dim() } , target dimension is: ${ target.dim() }");
        int curTarget = (int) target.valueAt(1);
        if (curTarget == paddingValue) {
          return gradInput;
        }

        double temp = -1;
        if (weights != null) {
          temp = TensorNumeric.times(-1, weights.valueAt(curTarget));
        }
        gradInput.setValue(curTarget, temp);

        if (sizeAverage) {
          gradInput.setValue(curTarget, TensorNumeric.divide(gradInput.valueAt(curTarget),
              total_weight));
        }
        if (!logProbAsInput) {
          double clipped = TensorNumeric.clip(input.valueAt(curTarget), epsilon, oneMinusEpsilon);
          gradInput.setValue(curTarget,
              TensorNumeric.times(gradInput.valueAt(curTarget), TensorNumeric.inv(clipped)));
        }
      } else if (input.dim() == 2) {
        int batchSize = input.size(1);
        int[] targetSize = target.size();
        target.squeeze();

        for (int i = 1; i <= batchSize; i++) {
          int curTarget = (int) target.valueAt(i);
          if (curTarget != paddingValue) {
            double temp = -1;
            if (weights != null) {
              temp = TensorNumeric.times(-1, weights.valueAt(curTarget));
            }

            gradInput.setValue(i, curTarget, temp);
            if (sizeAverage) {
              gradInput.setValue(i, curTarget, TensorNumeric.divide(gradInput.valueAt(i,
                  curTarget), total_weight));
            }
            if (!logProbAsInput) {
              double clipped = TensorNumeric.clip(input.valueAt(i, curTarget),
                  epsilon, oneMinusEpsilon);
              gradInput.setValue(i, curTarget,
                  TensorNumeric.times(gradInput.valueAt(i, curTarget), TensorNumeric.inv(clipped)));
            }
          }
        }
        target.resize(targetSize);
      }
    }
    return gradInput;
  }
}
