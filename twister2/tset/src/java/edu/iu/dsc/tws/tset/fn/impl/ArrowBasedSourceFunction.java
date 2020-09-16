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
package edu.iu.dsc.tws.tset.fn.impl;

import java.io.Serializable;
import java.util.logging.Logger;

import org.apache.arrow.vector.IntVector;

import edu.iu.dsc.tws.api.tset.TSetContext;
import edu.iu.dsc.tws.api.tset.fn.BaseSourceFunc;
import edu.iu.dsc.tws.data.arrow.Twister2ArrowFileReader;

public class ArrowBasedSourceFunction extends BaseSourceFunc<Object> implements Serializable {

  private static final Logger LOG = Logger.getLogger(ArrowBasedSourceFunction.class.getName());

  private final String arrowInputDirectory;
  private final String arrowInputFile;
  private final String arrowSchema;

  private int parallel;
  private int currentCell = 0;
  private IntVector intVector = null;

  private transient Twister2ArrowFileReader twister2ArrowFileReader;

  public ArrowBasedSourceFunction(String arrowInputdirectory, String arrowinputFile,
                                  int parallelism, String schema) {
    this.arrowInputDirectory = arrowInputdirectory;
    this.arrowInputFile = arrowinputFile;
    this.parallel = parallelism;
    this.arrowSchema = schema;
  }

  /**
   * Prepare method
   */
  public void prepare(TSetContext context) {
    super.prepare(context);
    this.twister2ArrowFileReader = new Twister2ArrowFileReader(this.arrowInputDirectory
        + "/" + context.getWorkerId() + "/" + this.arrowInputFile, arrowSchema);
    this.twister2ArrowFileReader.initInputFile();
  }

  @Override
  public boolean hasNext() {
    try {
      if (intVector == null || currentCell == intVector.getValueCount()) {
        intVector = twister2ArrowFileReader.getIntegerVector();
        currentCell = 0;
      }
      return intVector != null && currentCell < intVector.getValueCount();
    } catch (Exception e) {
      throw new RuntimeException("Unable to read int vector", e);
    }
  }

  @Override
  public Object next() {
    return intVector.get(currentCell++);
  }
}
