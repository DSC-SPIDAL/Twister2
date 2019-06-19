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
package edu.iu.dsc.tws.examples.ml.svm.data;

import java.util.Iterator;
import java.util.logging.Logger;

import edu.iu.dsc.tws.task.api.BaseCompute;
import edu.iu.dsc.tws.task.api.IMessage;

public class IterativeSVMWeightVectorObjectCompute extends BaseCompute {
  private static final Logger LOG = Logger.getLogger(IterativeSVMWeightVectorObjectCompute.class
      .getName());


  private static final long serialVersionUID = -254264120110286748L;

  /**
   * Edge name to write the partitoned datapoints
   */
  private String edgeName;

  /**
   * Task parallelism
   */
  private int parallelism;

  /**
   * Data size
   */
  private int datasize;

  /**
   * Dimension of the datapoints
   */
  private int features;

  /**
   * Datapoints array
   */
  private double[] dataPointsLocal;

  public IterativeSVMWeightVectorObjectCompute(String edgeName, int parallelism, int datasize,
                                               int features) {
    this.edgeName = edgeName;
    this.parallelism = parallelism;
    this.datasize = datasize;
    this.features = features;
    this.dataPointsLocal = new double[this.features];
  }

  public IterativeSVMWeightVectorObjectCompute(String edgeName, int datasize, int features) {
    this.edgeName = edgeName;
    this.datasize = datasize;
    this.features = features;
    this.dataPointsLocal = new double[this.features];
  }

  public String getEdgeName() {
    return edgeName;
  }

  public void setEdgeName(String edgeName) {
    this.edgeName = edgeName;
  }

  public int getParallelism() {
    return parallelism;
  }

  public void setParallelism(int parallelism) {
    this.parallelism = parallelism;
  }

  public int getDatasize() {
    return datasize;
  }

  public void setDatasize(int datasize) {
    this.datasize = datasize;
  }

  public int getFeatures() {
    return features;
  }

  public void setFeatures(int features) {
    this.features = features;
  }

  @Override
  public boolean execute(IMessage message) {
    Object o = message.getContent();
    if (o != null) {
      Iterator<?> itr = (Iterator) o;
      if (itr.hasNext()) {
        String s = (String) itr.next();
        String[] splits = s.split(",");
        if (features == splits.length) {
          for (int i = 0; i < splits.length; i++) {
            this.dataPointsLocal[i] = Double.parseDouble(splits[i]);
          }
        } else {
          LOG.severe(String.format("The weight vector and feature size doesn't match"));
        }
      }
      context.writeEnd(getEdgeName(), dataPointsLocal);
    } else {
      LOG.info(String.format("Something Went Wrong <Null Message>"));
    }

    return true;
  }
}
