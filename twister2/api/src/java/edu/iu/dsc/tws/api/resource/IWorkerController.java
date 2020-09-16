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

package edu.iu.dsc.tws.api.resource;

import java.util.List;

import edu.iu.dsc.tws.api.checkpointing.CheckpointingClient;
import edu.iu.dsc.tws.api.exceptions.JobFaultyException;
import edu.iu.dsc.tws.api.exceptions.TimeoutException;
import edu.iu.dsc.tws.proto.jobmaster.JobMasterAPI;

/**
 * an interface to get the list of workers in a job and their addresses
 */
public interface IWorkerController {

  /**
   * return the WorkerInfo object for this worker
   */
  JobMasterAPI.WorkerInfo getWorkerInfo();

  /**
   * return the WorkerInfo object for the given ID
   */
  JobMasterAPI.WorkerInfo getWorkerInfoForID(int id);

  /**
   * return the number of all workers in this job,
   * including non-started ones and finished ones
   */
  int getNumberOfWorkers();

  /**
   * get worker restartCount
   * zero means starting for the first time
   */
  int workerRestartCount();

  /**
   * get all joined workers in this job, including the ones finished execution
   * if there are some workers that have not joined yet, they may not be included in this list.
   * users can compare the total number of workers to the size of this list and
   * understand whether there are non-joined workers
   */
  List<JobMasterAPI.WorkerInfo> getJoinedWorkers();

  /**
   * get all workers in the job.
   * If some workers has not joined the job yet, wait for them.
   * After waiting for the timeout specified in ControllerContext.maxWaitTimeForAllToJoin
   * if some workers still could not join, throw an exception
   * <p>
   * return all workers in the job including the ones that have already left, if any
   */
  List<JobMasterAPI.WorkerInfo> getAllWorkers() throws TimeoutException;

  /**
   * wait for all workers in the job to arrive at this barrier
   * After waiting for the timeout specified in ControllerContext.maxWaitTimeOnBarrier
   * if some workers still could not arrive at the barrier, throw an exception
   */
  void waitOnBarrier() throws TimeoutException, JobFaultyException;

  /**
   * wait for all workers in the job to arrive at this barrier
   * After waiting for the timeLimit,
   * if some workers still could not arrive at the barrier, throw an exception
   */
  void waitOnBarrier(long timeLimit) throws TimeoutException, JobFaultyException;

  /**
   * this barrier is used when initializing the workers.
   * when there is a failure in the job,
   * workers synchronize with this barrier
   * this must not be used in other parts of the system
   */
  void waitOnInitBarrier() throws TimeoutException;

  /**
   * Get the failure listener
   *
   * @return the failure listener
   */
  default IWorkerFailureListener getFailureListener() {
    return null;
  }

  default CheckpointingClient getCheckpointingClient() {
    //todo remove null return
    return null;
  }
}
