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
package edu.iu.dsc.tws.task.graph;

import java.util.HashSet;
import java.util.Set;
import java.util.TreeSet;
import java.util.logging.Logger;

import edu.iu.dsc.tws.comms.api.DataFlowOperation;
import edu.iu.dsc.tws.task.api.ITask;
import edu.iu.dsc.tws.task.api.Task;
import edu.iu.dsc.tws.task.taskgraphfluentapi.ITaskInfo;

/**
 * This is the main class for creating the dataflow task graph.
 */
public class DataflowTaskGraphGenerator {
  private static final Logger LOG = Logger.getLogger(DataflowTaskGraphGenerator.class.getName());

  /**
   * Newly added code for defining the task edges as dataflow operations namely
   * Map, Reduce, Shuffle, and others.
   */
  private ITaskGraph<ITask, Edge> taskgraph =
      new BaseDataflowTaskGraph<>();
  private ITaskGraph<TaskGraphMapper, Edge> tGraph =
      new BaseDataflowTaskGraph<>();
  private ITaskGraph<ITaskInfo, Edge> iTaskGraph =
      new BaseDataflowTaskGraph<>();
  private ITaskGraph<TaskMapper, CManager> dataflowTaskGraph =
      new BaseDataflowTaskGraph<>();

  private Set<SourceTargetTaskDetails> sourceTargetTaskDetailsSet = new TreeSet<>();

  public Set<SourceTargetTaskDetails> getSourceTargetTaskDetailsSet() {
    return sourceTargetTaskDetailsSet;
  }

  public void setSourceTargetTaskDetailsSet(Set<SourceTargetTaskDetails>
                                                sourceTargetTaskDetailsSet) {
    this.sourceTargetTaskDetailsSet = sourceTargetTaskDetailsSet;
  }

  //Newly Added on April 5th, 2018
  private ITaskGraph<Task, DataFlowOperation> dataflowGraph =
      new BaseDataflowTaskGraph<>();

  public ITaskGraph<ITask, Edge> getTaskgraph() {
    return taskgraph;
  }

  public void setTaskgraph(ITaskGraph<ITask,
      Edge> taskgraph) {
    this.taskgraph = taskgraph;
  }

  public DataflowTaskGraphGenerator generateTaskGraph(ITask task1,
                                                      ITask task2,
                                                      Edge... dataflowOperation) {
    try {
      this.taskgraph.addTaskVertex(task1);
      this.taskgraph.addTaskVertex(task2);
      this.taskgraph.addTaskEdge(task1, task2, dataflowOperation[0]);
    } catch (IllegalArgumentException iae) {
      iae.printStackTrace();
    }
    return this;
  }

  public Set<SourceTargetTaskDetails> getAllParentTasks(String taskName) {

    Set<SourceTargetTaskDetails> allParentTaskDetailsSet = new TreeSet<>();
    if (!getSourceTargetTaskDetailsSet().isEmpty()) {
      for (SourceTargetTaskDetails taskDetails : this.getSourceTargetTaskDetailsSet()) {
        allParentTaskDetailsSet.add(taskDetails);
      }
    }
    /*Set<SourceTargetTaskDetails> filtered = sourceTargetTaskDetailsSet.stream()
        .filter(mc -> mc.getTargetTask().taskName().equalsIgnoreCase("task1"))
        .collect(Collectors.toSet());
    filtered.forEach(mc -> System.out.println("Object: " + mc.getTargetTask().taskName()));*/

    if (!allParentTaskDetailsSet.isEmpty()) {
      System.out.println("Parent Task Details Size:0" + allParentTaskDetailsSet.size());
    }
    return allParentTaskDetailsSet;
  }

  public Set<SourceTargetTaskDetails> getDataflowTaskChildTasks() {
    final ITaskGraph<ITask, Edge> dataflowTaskgraph = this.getTaskgraph();
    Set<ITask> taskVertices = dataflowTaskgraph.getTaskVertexSet();
    //Newly Added on April 5th, 2018
    //Set<SourceTargetTaskDetails> sourceTargetTaskDetailsSet = new HashSet<>();
    for (ITask child : taskVertices) {
      sourceTargetTaskDetailsSet = dataflowTaskSourceTargetVertices(dataflowTaskgraph, child);
      if (!sourceTargetTaskDetailsSet.isEmpty()) {
        for (SourceTargetTaskDetails sourceTargetTaskDetails : sourceTargetTaskDetailsSet) {
          LOG.info("Source and Target Task Details:"
              + sourceTargetTaskDetails.getSourceTask() + "--->"
              + sourceTargetTaskDetails.getTargetTask() + "---"
              + "Source Task Id and Name" + "---"
              + sourceTargetTaskDetails.getSourceTask().taskName() + "----"
              + "Target Task Id and Name" + "---"
              + sourceTargetTaskDetails.getTargetTask().taskName() + "---"
              + sourceTargetTaskDetails.getDataflowOperationName() + "\n");
        }
      }
    }
    setSourceTargetTaskDetailsSet(sourceTargetTaskDetailsSet);
    return sourceTargetTaskDetailsSet;
  }

  //Newly Added on April 5th, 2018
  public Set<SourceTargetTaskDetails> getDataflowTaskChildTasks(
      DataflowTaskGraphGenerator taskGraph1) {

    final ITaskGraph<ITask, Edge> dataflowTaskgraph = taskGraph1.
        getTaskgraph();
    Set<ITask> taskVertices = dataflowTaskgraph.getTaskVertexSet();

    //Newly Added on April 5th, 2018
    //Set<SourceTargetTaskDetails> sourceTargetTaskDetailsSet = new HashSet<>();
    for (ITask child : taskVertices) {
      sourceTargetTaskDetailsSet = dataflowTaskSourceTargetVertices(dataflowTaskgraph, child);
      if (!sourceTargetTaskDetailsSet.isEmpty()) {
        for (SourceTargetTaskDetails sourceTargetTaskDetails : sourceTargetTaskDetailsSet) {
          LOG.info("Source and Target Task Details:"
              + sourceTargetTaskDetails.getSourceTask() + "--->"
              + sourceTargetTaskDetails.getTargetTask() + "---"
              + "Source Task Id and Name" + "---"
              + sourceTargetTaskDetails.getSourceTask().taskName() + "----"
              + "Target Task Id and Name" + "---"
              + sourceTargetTaskDetails.getTargetTask().taskName() + "---"
              + sourceTargetTaskDetails.getDataflowOperationName() + "\n");
        }
      }
    }
    setSourceTargetTaskDetailsSet(sourceTargetTaskDetailsSet);
    return sourceTargetTaskDetailsSet;
  }

  /**
   * This method displays the task edges, its child, source and target task vertices of a particular
   * task
   */
  private Set<SourceTargetTaskDetails> dataflowTaskSourceTargetVertices(
      final ITaskGraph<ITask,
          Edge> dataflowtaskgraph,
      final ITask mapper) {

    /*LOG.info("Task Object is:" + mapper + "\t"
         + "Task Id:" + mapper.getTaskId() + "\t"
        + "Task Name:" + mapper.taskName());*/

    Set<SourceTargetTaskDetails> childTask = new HashSet<>();
     /*if (dataflowtaskgraph.outDegreeOfTask(mapper) == 0) {
      return childTask;
    } else {
      Set<TaskEdge> taskEdgesOf = dataflowtaskgraph.outgoingTaskEdgesOf(mapper);
      for (TaskEdge edge : taskEdgesOf) {
        SourceTargetTaskDetails sourceTargetTaskDetails = new SourceTargetTaskDetails();
        sourceTargetTaskDetails.setSourceTask(dataflowtaskgraph.getTaskEdgeSource(edge));
        sourceTargetTaskDetails.setTargetTask(dataflowtaskgraph.getTaskEdgeTarget(edge));
        sourceTargetTaskDetails.setDataflowOperation(edge);
        sourceTargetTaskDetails.setDataflowOperationName(edge.getDataflowOperation());
        childTask.add(sourceTargetTaskDetails);

        LOG.info("%%%% Dataflow Operation:" + edge.getDataflowOperation());
        LOG.info("%%%% Source and Target Vertex:" + dataflowtaskgraph.getTaskEdgeSource(edge)
            + "\t" + dataflowtaskgraph.getTaskEdgeTarget(edge));
      }
      return childTask;
    }*/

    Set<Edge> edgesOf = dataflowtaskgraph.outgoingTaskEdgesOf(mapper);
    for (Edge edge : edgesOf) {
      SourceTargetTaskDetails sourceTargetTaskDetails = new SourceTargetTaskDetails();
      sourceTargetTaskDetails.setDataflowOperation(edge);
      sourceTargetTaskDetails.setDataflowOperationName(edge.name());
      childTask.add(sourceTargetTaskDetails);
    }
    return childTask;
  }

  private ITaskGraph<TaskMapper, DataflowTaskEdge> taskGraph =
      new BaseDataflowTaskGraph<>();

  public ITaskGraph<TaskMapper, DataflowTaskEdge> getTaskGraph() {
    return taskGraph;
  }

  public void setTaskGraph(ITaskGraph<TaskMapper,
      DataflowTaskEdge> taskGraph) {
    this.taskGraph = taskGraph;
  }

  public ITaskGraph<TaskMapper, CManager> getDataflowTaskGraph() {
    return dataflowTaskGraph;
  }

  public void setDataflowTaskGraph(ITaskGraph<TaskMapper,
      CManager> dataflowTaskGraph) {
    this.dataflowTaskGraph = dataflowTaskGraph;
  }

  public ITaskGraph<ITaskInfo, Edge> getITaskGraph() {
    return iTaskGraph;
  }

  public void setITaskGraph(ITaskGraph<ITaskInfo, Edge> iTaskgraph) {
    this.iTaskGraph = iTaskgraph;
  }

  public ITaskGraph<TaskGraphMapper, Edge> getTGraph() {
    return tGraph;
  }

  public void setTGraph(ITaskGraph<TaskGraphMapper, Edge> tgraph) {
    this.tGraph = tgraph;
  }

  /**
   * This method is responsible for creating the dataflow task graph from the receiving
   * task vertices and task eges.
   */
  public DataflowTaskGraphGenerator generateITaskGraph(
      Edge dataflowOperation,
      ITaskInfo taskVertex, ITaskInfo... taskEdge) {
    try {
      this.iTaskGraph.addTaskVertex(taskVertex);
      if (taskEdge.length >= 1) {
        this.iTaskGraph.addTaskVertex(taskEdge[0]);
        this.iTaskGraph.addTaskEdge(taskVertex, taskEdge[0], dataflowOperation);
      }
    } catch (IllegalArgumentException iae) {
      iae.printStackTrace();
    }
    System.out.println("Constructed Task Graph is:" + iTaskGraph.getTaskVertexSet().size());
    return this;
  }


  public DataflowTaskGraphGenerator generateTGraph(TaskGraphMapper sourceTask,
                                                   TaskGraphMapper sinkTask,
                                                   Edge... dataflowOperation) {
    try {
      this.tGraph.addTaskVertex(sourceTask);
      this.tGraph.addTaskVertex(sinkTask);
      this.tGraph.addTaskEdge(sourceTask, sinkTask, dataflowOperation[0]);
    } catch (IllegalArgumentException iae) {
      iae.printStackTrace();
    }
    return this;
  }

  public DataflowTaskGraphGenerator generateTGraph(TaskGraphMapper taskGraphMapper1,
                                                   TaskGraphMapper... taskGraphMappers) {
    try {
      this.tGraph.addTaskVertex(taskGraphMapper1);
      for (TaskGraphMapper mapperTask : taskGraphMappers) {
        this.tGraph.addTaskEdge(mapperTask, taskGraphMapper1);
      }
    } catch (IllegalArgumentException iae) {
      iae.printStackTrace();
    }
    return this;
  }


  public DataflowTaskGraphGenerator generateTaskGraph(ITask sourceTask,
                                                      ITask... sinkTask) {
    try {
      this.taskgraph.addTaskVertex(sourceTask);
      for (ITask mapperTask : sinkTask) {
        this.taskgraph.addTaskEdge(mapperTask, sourceTask);
      }
    } catch (IllegalArgumentException iae) {
      iae.printStackTrace();
    }
    return this;
  }

  public DataflowTaskGraphGenerator generateDataflowTaskGraph(TaskMapper taskMapperTask1,
                                                              TaskMapper taskMapperTask2,
                                                              CManager... cManagerTask) {
    try {
      this.dataflowTaskGraph.addTaskVertex(taskMapperTask1);
      this.dataflowTaskGraph.addTaskVertex(taskMapperTask2);
      for (CManager cManagerTask1 : cManagerTask) {
        this.dataflowTaskGraph.addTaskEdge(
            taskMapperTask1, taskMapperTask2, cManagerTask[0]);
      }
    } catch (IllegalArgumentException iae) {
      iae.printStackTrace();
    }
    LOG.info("Generated Dataflow Task Graph Is:" + taskGraph);
    return this;
  }


  public void removeTaskVertex(TaskGraphMapper mapperTask) {
    LOG.info("Mapper task done to be removed:" + mapperTask);
    this.tGraph.removeTaskVertex(mapperTask);
    LOG.info("Now the task graph is:" + this.dataflowTaskGraph);
  }

  public void removeTaskVertex(ITask mapperTask) {
    LOG.info("Mapper task done to be removed:" + mapperTask);
    this.taskgraph.removeTaskVertex(mapperTask);
    LOG.info("Now the task graph is:" + this.dataflowTaskGraph);
  }


  public DataflowTaskGraphGenerator generateDataflowGraph(Task sourceTask,
                                                          Task sinkTask,
                                                          DataFlowOperation... dataFlowOperation) {
    try {
      this.dataflowGraph.addTaskVertex(sourceTask);
      this.dataflowGraph.addTaskVertex(sinkTask);
      for (DataFlowOperation dataflowOperation1 : dataFlowOperation) {
        this.dataflowGraph.addTaskEdge(
            sourceTask, sinkTask, dataFlowOperation[0]);
      }
    } catch (IllegalArgumentException iae) {
      iae.printStackTrace();
    }
    LOG.info("Generated Dataflow Task Graph Is:" + taskGraph);
    return this;
  }
}

