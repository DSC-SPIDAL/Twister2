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
package edu.iu.dsc.tws.task.dataobjects;

import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.compute.TaskContext;
import edu.iu.dsc.tws.api.compute.executor.ExecutorContext;
import edu.iu.dsc.tws.api.compute.nodes.BaseSource;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.data.Path;
import edu.iu.dsc.tws.data.api.formatters.LocalCSVInputPartitioner;
import edu.iu.dsc.tws.data.fs.io.InputSplit;
import edu.iu.dsc.tws.dataset.DataSource;
import edu.iu.dsc.tws.executor.core.ExecutionRuntime;

public class DataObjectCSVSource extends BaseSource {

  private static final Logger LOG = Logger.getLogger(DataObjectCSVSource.class.getName());

  private static final long serialVersionUID = -1L;

  private DataSource<?, ?> source;

  private String edgeName;
  private String dataDirectory;

  public DataObjectCSVSource() {
  }

  public DataObjectCSVSource(String edgename, String dataDirectory) {
    this.edgeName = edgename;
    this.dataDirectory = dataDirectory;
  }

  public String getDataDirectory() {
    return dataDirectory;
  }

  public void setDataDirectory(String dataDirectory) {
    this.dataDirectory = dataDirectory;
  }

  /**
   * Getter property to set the edge name
   */
  public String getEdgeName() {
    return edgeName;
  }

  /**
   * Setter property to set the edge name
   */
  public void setEdgeName(String edgeName) {
    this.edgeName = edgeName;
  }


  @Override
  public void execute() {
    InputSplit<?> inputSplit = source.getNextSplit(context.taskIndex());
    while (inputSplit != null) {
      try {
        while (!inputSplit.reachedEnd()) {
          Object value = inputSplit.nextRecord(null);
          LOG.info("object values are:" + value);
          if (value != null) {
            context.write(getEdgeName(), value);
          }
        }
        inputSplit = source.getNextSplit(context.taskIndex());
      } catch (IOException e) {
        LOG.log(Level.SEVERE, "Failed to read the input", e);
      }
    }
    context.end(getEdgeName());
  }

  @Override
  public void prepare(Config cfg, TaskContext context) {
    super.prepare(cfg, context);
    ExecutionRuntime runtime = (ExecutionRuntime) cfg.get(ExecutorContext.TWISTER2_RUNTIME_OBJECT);
    this.source = runtime.createInput(cfg, context, new LocalCSVInputPartitioner(
        new Path(getDataDirectory()), context.getParallelism(), cfg));
  }
}
