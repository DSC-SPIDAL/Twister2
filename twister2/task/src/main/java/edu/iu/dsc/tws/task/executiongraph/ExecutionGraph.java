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
package edu.iu.dsc.tws.task.executiongraph;

import java.util.ArrayList;
import java.util.Set;
import java.util.logging.Logger;

import edu.iu.dsc.tws.task.api.ITask;
import edu.iu.dsc.tws.task.core.TaskExecutorFixedThread;
import edu.iu.dsc.tws.task.graph.DataflowTaskGraphGenerator;
import edu.iu.dsc.tws.task.graph.DataflowTaskGraphParser;
import edu.iu.dsc.tws.task.graph.TaskExecutor;

/**
 * This is the simple execution graph generator and it will be extended
 * further by dynamically identifying (container values) and
 * generating the execution graph...!
 */

public class ExecutionGraph implements IExecutionGraph {

  private static final Logger LOG = Logger.getLogger(ExecutionGraph.class.getName());

  private Set<ITask> parsedTaskSet;
  private DataflowTaskGraphParser dataflowTaskGraphParser = null;
  //private DataflowTGraphParser dataflowTGraphParser = null;
  private TaskExecutorFixedThread taskExecutor;

  /**
   * Constructor to initialize the parsed task graph set which has all
   * the task vertices and edges.
   */
  public ExecutionGraph(Set<ITask> parsedTaskSet) {
    this.parsedTaskSet = parsedTaskSet;
  }

  /**
   * This method will not be used in future, it will be replaced with an
   * execution graph parser if it requires.
   */
  public Set<ITask> parseTaskGraph(DataflowTaskGraphGenerator
                                       dataflowTaskGraphGenerator) {
    if (dataflowTaskGraphGenerator != null) {
      //dataflowTGraphParser = new DataflowTGraphParser(dataflowTaskGraphGenerator);
      //parsedTaskSet = dataflowTGraphParser.dataflowTGraphParseAndSchedule();
      dataflowTaskGraphParser = new DataflowTaskGraphParser(dataflowTaskGraphGenerator);
      parsedTaskSet = dataflowTaskGraphParser.taskGraphParseAndSchedule();
    }
    return parsedTaskSet;
  }

  /**
   * This method is responsible for generating the execution graph to be executed by
   * the executors which will receive only the containerId as an input...!
   */
  public String generateExecutionGraph(int containerId) {
    //public TaskExecutor generateExecutionGraph(int containerId){

    //For testing purpose and it will replaced with actual task executor...!
    TaskExecutor taskExecutionGraph = new TaskExecutor();

    if (!parsedTaskSet.isEmpty() && !(containerId < 0)) {
      if (containerId == 0) {
        //taskExecutor.registerTask(parsedTaskSet.iterator().next());
        //taskExecutor.submitTask(0);
        //taskExecutor.progres();

        //For testing purpose...!
        //taskExecutionGraph.execute(parsedTaskSet.iterator().next());
      } else if (containerId >= 1) { //This loop should be modified for the complex task graphs
        int index = 0;
        for (ITask processedTask : parsedTaskSet) {
          if (index == 0) {
            ++index;
          } else if (index == 1) {
            ArrayList<Integer> inq = new ArrayList<>();
            inq.add(0);
            //taskExecutor.setTaskMessageProcessLimit(10000);
            //taskExecutor.registerSinkTask(processedTask, inq);//enabled latter
            //taskExecutor.progres();

            //For testing purpose...! It is working...but not properly.
            //taskExecutionGraph.execute(processedTask);
            ++index;
          } else if (index == 2) {
            ArrayList<Integer> inq1 = new ArrayList<>();
            inq1.add(0);
            //taskExecutor.setTaskMessageProcessLimit(10000);
            //taskExecutor.registerSinkTask(processedTask, inq1);//enabled latter
            //taskExecutor.progres();

            //For testing purpose...!
            //taskExecutionGraph.execute(processedTask);
            ++index;
          } else if (index == 3) {
            ArrayList<Integer> inq1 = new ArrayList<>();
            inq1.add(1);
            inq1.add(2);
            //taskExecutor.setTaskMessageProcessLimit(10000);
            //taskExecutor.registerSinkTask(processedTask, inq1);//enabled latter
            //taskExecutor.progres();

            //For testing purpose...!
            //taskExecutionGraph.execute(processedTask);
            //LOG.info("Container 3 task is:" + processedTask.getInputData());
            ++index;
          } else if (index > 3) {
            //it would be constructed based on the container value and no.of tasks
            LOG.info("Task Index is greater than 3");
            //break;
          }
        }
      }
    }
    return "Generated Execution Graph Successfully";
    //return taskExecutionGraph;
  }

  /**
   * This method is responsible for generating the execution graph to be executed by
   * the executors...!
   */
  public String generateExecutionGraph(int containerId, Set<ITask> processedTaskSet) {
    //public TaskExecutor generateExecutionGraph(int containerId,
    //                                           Set<TaskGraphMapper> processedTaskSet) {

    //For testing purpose and it will replaced with actual task executor...!
    //TaskExecutor taskExecutionGraph = new TaskExecutor();

    if (!parsedTaskSet.isEmpty() && !(containerId < 0)) {
      if (containerId == 0) {
        //taskExecutor.registerTask(parsedTaskSet.iterator().next());
        //taskExecutor.submitTask(0);
        //taskExecutor.progres();

        //For testing purpose...!
        //taskExecute.execute(parsedTaskSet.iterator().next());
      } else if (containerId >= 1) { //This loop should be modified for the complex task graphs
        int index = 0;
        LOG.info("%%%%%%% Parsed Task Set Size Is: %%%%%" + parsedTaskSet.size());
        for (ITask processedTask : parsedTaskSet) {
          if (index == 0) {
            ++index;
          } else if (index == 1) {
            ArrayList<Integer> inq = new ArrayList<>();
            inq.add(0);
            //taskExecutor.setTaskMessageProcessLimit(10000);
            //taskExecutor.registerSinkTask(processedTask, inq);//enabled latter
            //taskExecutor.progres();

            //For testing purpose...! It is working...but not properly.
            //taskExecute.execute(processedTask);
            ++index;
          } else if (index == 2) {
            ArrayList<Integer> inq1 = new ArrayList<>();
            inq1.add(0);
            //taskExecutor.setTaskMessageProcessLimit(10000);
            //taskExecutor.registerSinkTask(processedTask, inq1);//enabled latter
            //taskExecutor.progres();

            //For testing purpose...!
            //taskExecute.execute(processedTask);
            ++index;
          } else if (index == 3) {
            ArrayList<Integer> inq1 = new ArrayList<>();
            inq1.add(1);
            inq1.add(2);
            //taskExecutor.setTaskMessageProcessLimit(10000);
            //taskExecutor.registerSinkTask(processedTask, inq1);//enabled latter
            //taskExecutor.progres();

            //For testing purpose...!
            //taskExecute.execute(processedTask);
            ++index;
          } else if (index > 3) {
            //it would be constructed based on the container value and no.of tasks
            LOG.info("Task Index is greater than 3");
            //break;
          }
        }
      }
    }
    return "Generated Execution Graph Successfully";
    //return taskExecutionGraph;
  }
}



