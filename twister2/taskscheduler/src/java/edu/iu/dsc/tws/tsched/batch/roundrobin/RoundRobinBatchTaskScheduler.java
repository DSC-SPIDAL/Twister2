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
package edu.iu.dsc.tws.tsched.batch.roundrobin;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.logging.Logger;

import edu.iu.dsc.tws.api.compute.exceptions.TaskSchedulerException;
import edu.iu.dsc.tws.api.compute.graph.ComputeGraph;
import edu.iu.dsc.tws.api.compute.graph.Vertex;
import edu.iu.dsc.tws.api.compute.schedule.ITaskScheduler;
import edu.iu.dsc.tws.api.compute.schedule.elements.Resource;
import edu.iu.dsc.tws.api.compute.schedule.elements.TaskInstanceId;
import edu.iu.dsc.tws.api.compute.schedule.elements.TaskInstancePlan;
import edu.iu.dsc.tws.api.compute.schedule.elements.TaskSchedulePlan;
import edu.iu.dsc.tws.api.compute.schedule.elements.Worker;
import edu.iu.dsc.tws.api.compute.schedule.elements.WorkerPlan;
import edu.iu.dsc.tws.api.compute.schedule.elements.WorkerSchedulePlan;
import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.tsched.spi.common.TaskSchedulerContext;
import edu.iu.dsc.tws.tsched.spi.taskschedule.TaskInstanceMapCalculation;
import edu.iu.dsc.tws.tsched.utils.TaskAttributes;
import edu.iu.dsc.tws.tsched.utils.TaskVertexParser;

/**
 * This class allocate the task instances into the container in a round robin manner.
 * First, it will allocate the task instances into the logical container values and then
 * it will calculate the required ram, disk, and cpu values for task instances and the logical
 * containers which is based on the task configuration values and the allocated work values
 * respectively.
 * <p>
 * {@literal For example, if there are two tasks with parallelism value of 2, 1st task -> instance 0 will
 * go to container 0, 2nd task -> instance 0 will go to container 1, 1st task -> instance 1 will
 * go to container 1, 2nd task -> instance 1 will go to container 1.}
 */

public class RoundRobinBatchTaskScheduler implements ITaskScheduler {

  private static final Logger LOG = Logger.getLogger(RoundRobinBatchTaskScheduler.class.getName());

  //Represents global task Id
  private int gTaskId = 0;

  //Represents the task instance ram
  private Double instanceRAM;

  //Represents the task instance disk
  private Double instanceDisk;

  //Represents the task instance cpu value
  private Double instanceCPU;

  //Config object
  private Config config;

  //WorkerId
  private int workerId;

  //Round Robin Allocation Map
  private Map<Integer, List<TaskInstanceId>> roundRobinAllocation;

  //Task Attributes Object
  private TaskAttributes taskAttributes;

  /**
   * This method initialize the task instance values with the values specified in the task config
   * object.
   */
  @Override
  public void initialize(Config cfg) {
    this.config = cfg;
    this.instanceRAM = TaskSchedulerContext.taskInstanceRam(config);
    this.instanceDisk = TaskSchedulerContext.taskInstanceDisk(config);
    this.instanceCPU = TaskSchedulerContext.taskInstanceCpu(config);
    this.roundRobinAllocation = new HashMap<>();
    this.taskAttributes = new TaskAttributes();
  }

  @Override
  public void initialize(Config cfg, int workerid) {
    this.initialize(cfg);
    this.workerId = workerid;
  }

  /**
   * This is the base method which receives the dataflow taskgraph and the worker plan to allocate
   * the task instances to the appropriate workers with their required ram, disk, and cpu values.
   */
  @Override
  public TaskSchedulePlan schedule(ComputeGraph computeGraph, WorkerPlan workerPlan) {

    Map<Integer, List<TaskInstanceId>> containerInstanceMap;
    Map<Integer, WorkerSchedulePlan> containerPlans = new LinkedHashMap<>();

    for (int i = 0; i < workerPlan.getNumberOfWorkers(); i++) {
      roundRobinAllocation.put(i, new ArrayList<>());
    }

    //To retrieve the batch task instances(it may be single task vertex or a batch of task vertices)
    Set<Vertex> taskVertexSet = new LinkedHashSet<>(computeGraph.getTaskVertexSet());
    TaskVertexParser taskGraphParser = new TaskVertexParser();
    List<Set<Vertex>> taskVertexList = taskGraphParser.parseVertexSet(computeGraph);

    for (Set<Vertex> vertexSet : taskVertexList) {
      if (vertexSet.size() > 1) {
        containerInstanceMap = roundRobinBatchSchedulingAlgorithm(computeGraph, vertexSet);
      } else {
        Vertex vertex = vertexSet.iterator().next();
        containerInstanceMap = roundRobinBatchSchedulingAlgorithm(computeGraph, vertex);
      }

      TaskInstanceMapCalculation instanceMapCalculation =
          new TaskInstanceMapCalculation(this.instanceRAM, this.instanceCPU, this.instanceDisk);

      Map<Integer, Map<TaskInstanceId, Double>> instancesRamMap =
          instanceMapCalculation.getInstancesRamMapInContainer(containerInstanceMap,
              taskVertexSet);
      Map<Integer, Map<TaskInstanceId, Double>> instancesDiskMap =
          instanceMapCalculation.getInstancesDiskMapInContainer(containerInstanceMap,
              taskVertexSet);
      Map<Integer, Map<TaskInstanceId, Double>> instancesCPUMap =
          instanceMapCalculation.getInstancesCPUMapInContainer(containerInstanceMap,
              taskVertexSet);

      for (int containerId : containerInstanceMap.keySet()) {

        double containerRAMValue = TaskSchedulerContext.containerRamPadding(config);
        double containerDiskValue = TaskSchedulerContext.containerDiskPadding(config);
        double containerCpuValue = TaskSchedulerContext.containerCpuPadding(config);

        List<TaskInstanceId> taskTaskInstanceIds = containerInstanceMap.get(containerId);
        Map<TaskInstanceId, TaskInstancePlan> taskInstancePlanMap = new HashMap<>();

        for (TaskInstanceId id : taskTaskInstanceIds) {

          double instanceRAMValue = instancesRamMap.get(containerId).get(id);
          double instanceDiskValue = instancesDiskMap.get(containerId).get(id);
          double instanceCPUValue = instancesCPUMap.get(containerId).get(id);

          Resource instanceResource = new Resource(instanceRAMValue, instanceDiskValue,
              instanceCPUValue);

          taskInstancePlanMap.put(id, new TaskInstancePlan(
              id.getTaskName(), id.getTaskId(), id.getTaskIndex(), instanceResource));

          containerRAMValue += instanceRAMValue;
          containerDiskValue += instanceDiskValue;
          containerCpuValue += instanceDiskValue;
        }

        Worker worker = workerPlan.getWorker(containerId);
        Resource containerResource;

        if (worker != null && worker.getCpu() > 0
            && worker.getDisk() > 0 && worker.getRam() > 0) {
          containerResource = new Resource((double) worker.getRam(),
              (double) worker.getDisk(), (double) worker.getCpu());
        } else {
          containerResource = new Resource(containerRAMValue, containerDiskValue,
              containerCpuValue);
        }

        WorkerSchedulePlan taskWorkerSchedulePlan;
        if (containerPlans.containsKey(containerId)) {
          taskWorkerSchedulePlan = containerPlans.get(containerId);
          taskWorkerSchedulePlan.getTaskInstances().addAll(taskInstancePlanMap.values());
        } else {
          taskWorkerSchedulePlan = new WorkerSchedulePlan(
              containerId, new HashSet<>(taskInstancePlanMap.values()), containerResource);
          containerPlans.put(containerId, taskWorkerSchedulePlan);
        }
      }
    }
    return new TaskSchedulePlan(0, new HashSet<>(containerPlans.values()));
  }

  private Map<Integer, List<TaskInstanceId>> roundRobinBatchSchedulingAlgorithm(
      ComputeGraph graph, Vertex vertex) throws TaskSchedulerException {

    Map<String, Integer> parallelTaskMap;
    if (!graph.getGraphConstraints().isEmpty()) {
      if (!graph.getNodeConstraints().isEmpty()) {
        parallelTaskMap = taskAttributes.getParallelTaskMap(vertex, graph.getNodeConstraints());
      } else {
        parallelTaskMap = taskAttributes.getParallelTaskMap(vertex);
      }
      roundRobinAllocation = attributeBasedAllocation(parallelTaskMap, graph);
    } else {
      parallelTaskMap = taskAttributes.getParallelTaskMap(vertex);
      roundRobinAllocation = nonAttributeBasedAllocation(parallelTaskMap);
    }
    return roundRobinAllocation;
  }


  private Map<Integer, List<TaskInstanceId>> roundRobinBatchSchedulingAlgorithm(
      ComputeGraph graph, Set<Vertex> vertexSet) throws TaskSchedulerException {

    TreeSet<Vertex> orderedTaskSet = new TreeSet<>(new VertexComparator());
    orderedTaskSet.addAll(vertexSet);

    Map<String, Integer> parallelTaskMap;
    if (!graph.getGraphConstraints().isEmpty()) {
      if (!graph.getNodeConstraints().isEmpty()) {
        parallelTaskMap = taskAttributes.getParallelTaskMap(vertexSet,
            graph.getNodeConstraints());
      } else {
        parallelTaskMap = taskAttributes.getParallelTaskMap(vertexSet);
      }
      roundRobinAllocation = attributeBasedAllocation(parallelTaskMap, graph);
    } else {
      parallelTaskMap = taskAttributes.getParallelTaskMap(vertexSet);
      roundRobinAllocation = nonAttributeBasedAllocation(parallelTaskMap);
    }
    return roundRobinAllocation;
  }

  private Map<Integer, List<TaskInstanceId>> attributeBasedAllocation(
      Map<String, Integer> parallelTaskMap, ComputeGraph graph) {

    int containerIndex = 0;
    int instancesPerContainer = taskAttributes.getInstancesPerWorker(graph.getGraphConstraints());
    for (Map.Entry<String, Integer> e : parallelTaskMap.entrySet()) {
      String task = e.getKey();
      int taskParallelism = e.getValue();
      int numberOfInstances;
      if (instancesPerContainer < taskParallelism) {
        numberOfInstances = taskParallelism;
      } else {
        numberOfInstances = instancesPerContainer;
      }
      for (int taskIndex = 0; taskIndex < numberOfInstances; taskIndex++) {
        roundRobinAllocation.get(containerIndex).add(new TaskInstanceId(task, gTaskId, taskIndex));
        ++containerIndex;
        if (containerIndex >= roundRobinAllocation.size()) {
          containerIndex = 0;
        }
      }
      gTaskId++;
    }
    return roundRobinAllocation;
  }

  private Map<Integer, List<TaskInstanceId>> nonAttributeBasedAllocation(
      Map<String, Integer> parallelTaskMap) {

    int containerIndex = 0;
    for (Map.Entry<String, Integer> e : parallelTaskMap.entrySet()) {
      String task = e.getKey();
      int numberOfInstances = e.getValue();
      for (int taskIndex = 0; taskIndex < numberOfInstances; taskIndex++) {
        roundRobinAllocation.get(containerIndex).add(new TaskInstanceId(task, gTaskId, taskIndex));
        ++containerIndex;
        if (containerIndex >= roundRobinAllocation.size()) {
          containerIndex = 0;
        }
      }
      gTaskId++;
    }
    return roundRobinAllocation;
  }

  private static class VertexComparator implements Comparator<Vertex> {
    @Override
    public int compare(Vertex o1, Vertex o2) {
      return o1.getName().compareTo(o2.getName());
    }
  }
}

