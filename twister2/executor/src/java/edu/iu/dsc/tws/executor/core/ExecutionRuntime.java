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
package edu.iu.dsc.tws.executor.core;

import edu.iu.dsc.tws.api.comms.channel.TWSChannel;
import edu.iu.dsc.tws.api.compute.TaskContext;
import edu.iu.dsc.tws.api.compute.executor.ExecutionPlan;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.api.config.Context;
import edu.iu.dsc.tws.api.data.Path;
import edu.iu.dsc.tws.data.api.InputPartitioner;
import edu.iu.dsc.tws.data.fs.io.InputSplit;
import edu.iu.dsc.tws.dataset.DataSource;

/**
 * Captures the runtime information about the system.
 */
public class ExecutionRuntime {
  /**
   * Name of the job
   */
  private String jobName;

  /**
   * The job directory
   */
  private Path parentpath;

  /**
   * Execution plan
   */
  private ExecutionPlan plan;

  /**
   * The communication channel
   */
  private TWSChannel channel;

  public ExecutionRuntime(String jName, ExecutionPlan execPlan, TWSChannel ch) {
    this.jobName = jName;
    this.plan = execPlan;
    this.channel = ch;
  }

  public String getJobName() {
    return jobName;
  }

  public void setJobName(String jobName) {
    this.jobName = jobName;
  }

  public void setJobName(Config config) {
    this.setJobName(config.getStringValue(Context.JOB_NAME));
  }

  public TWSChannel getChannel() {
    return channel;
  }

  public ExecutionPlan getPlan() {
    return plan;
  }

  public Path getParentpath() {
    return parentpath;
  }

  public void setParentpath(Path parentpath) {
    this.parentpath = parentpath;
  }

  public <T, O extends InputSplit<T>> DataSource<T, O> createInput(
      Config cfg, TaskContext context, InputPartitioner<T, O> input) {
    return new DataSource<T, O>(cfg, input, context.getParallelism());
  }
}
