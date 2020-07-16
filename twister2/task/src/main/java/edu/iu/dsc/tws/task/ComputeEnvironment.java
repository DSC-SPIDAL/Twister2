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
package edu.iu.dsc.tws.task;

import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.compute.executor.ExecutionPlan;
import edu.iu.dsc.tws.api.compute.graph.ComputeGraph;
import edu.iu.dsc.tws.api.compute.graph.OperationMode;
import edu.iu.dsc.tws.api.exceptions.TimeoutException;
import edu.iu.dsc.tws.api.faulttolerance.JobProgress;
import edu.iu.dsc.tws.api.resource.WorkerEnvironment;
import edu.iu.dsc.tws.checkpointing.util.CheckpointingContext;
import edu.iu.dsc.tws.task.impl.ComputeGraphBuilder;
import edu.iu.dsc.tws.task.impl.TaskExecutor;

/**
 * Task environment.
 */
public final class ComputeEnvironment {
  private static final Logger LOG = Logger.getLogger(ComputeEnvironment.class.getName());

  /**
   * The worker environment
   */
  private WorkerEnvironment workerEnvironment;

  /**
   * The task environment
   */
  private TaskExecutor taskExecutor;

  /**
   * The task graph index
   */
  private static int taskGraphIndex = 0;

  /**
   * Singleton environment
   */
  private static volatile ComputeEnvironment computeEnv;

  private ComputeEnvironment(WorkerEnvironment workerEnv) {
    this.workerEnvironment = workerEnv;
    this.taskExecutor = new TaskExecutor(workerEnv);

    // if checkpointing enabled lets register for receiving faults
    if (CheckpointingContext.isCheckpointingEnabled(workerEnv.getConfig())) {
      JobProgress.registerFaultAcceptor(taskExecutor);
    }
  }

  /**
   * Use task executor for fine grained task graph manipulations. For single task graph builds,
   * use @buildAndExecute
   *
   * @return taskExecutor the current executor
   */
  public TaskExecutor getTaskExecutor() {
    return taskExecutor;
  }

  /**
   * for single task graph runs
   */
  public TaskExecutor buildAndExecute(ComputeGraphBuilder computeGraphBuilder) {
    ComputeGraph computeGraph = computeGraphBuilder.build();
    ExecutionPlan plan = this.getTaskExecutor().plan(computeGraph);
    this.getTaskExecutor().execute(computeGraph, plan);
    return this.getTaskExecutor();
  }

  /**
   * Create a compute graph builder
   *
   * @param operationMode specify the operation mode
   * @return the graph builder
   */
  public ComputeGraphBuilder newTaskGraph(OperationMode operationMode) {
    return this.newTaskGraph(operationMode, "task-graph-" + (taskGraphIndex++));
  }

  /**
   * Create a new task graph builder with a name
   *
   * @param operationMode operation node
   * @param name name of the graph
   * @return the graph builder
   */
  public ComputeGraphBuilder newTaskGraph(OperationMode operationMode, String name) {
    ComputeGraphBuilder computeGraphBuilder = ComputeGraphBuilder.newBuilder(
        workerEnvironment.getConfig());
    computeGraphBuilder.setMode(operationMode);
    computeGraphBuilder.setTaskGraphName(name);
    return computeGraphBuilder;
  }

  /**
   * Initialize the compute environment with the worker environment
   *
   * @param workerEnv worker environment
   * @return the compute environment
   */
  public static ComputeEnvironment init(WorkerEnvironment workerEnv) {
    if (computeEnv == null) {
      synchronized (ComputeEnvironment.class) {
        if (computeEnv == null) {
          computeEnv = new ComputeEnvironment(workerEnv);
        }
      }
    }
    return computeEnv;
  }

  /**
   * Closes the task environment
   * <p>
   * This method should be called at the end of worker
   */
  public void close() {
    try {
      LOG.info("Waiting on barrier in Compute Env...");
      workerEnvironment.getWorkerController().waitOnBarrier();
    } catch (TimeoutException timeoutException) {
      LOG.log(Level.SEVERE, timeoutException.getMessage(), timeoutException);
    }
    // if checkpointing enabled lets register for receiving faults
    if (CheckpointingContext.isCheckpointingEnabled(workerEnvironment.getConfig())) {
      JobProgress.unRegisterFaultAcceptor(taskExecutor);
    }
    // close the task executor
    taskExecutor.close();
    // close the worker environment
    workerEnvironment.close();
  }

  public Map<String, ExecutionPlan> build(ComputeGraph... computeGraphs) {
    return this.getTaskExecutor().plan(computeGraphs);
  }
}
