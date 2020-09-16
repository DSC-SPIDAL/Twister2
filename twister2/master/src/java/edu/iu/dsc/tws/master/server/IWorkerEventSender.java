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
package edu.iu.dsc.tws.master.server;

import edu.iu.dsc.tws.proto.jobmaster.JobMasterAPI;

public interface IWorkerEventSender {

  /**
   * the worker with the provided workerID failed
   * @param workerID
   */
  void workerFailed(int workerID);

  /**
   * the worker with the provided workerInfo restarted
   * @param workerInfo
   */
  void workerRestarted(JobMasterAPI.WorkerInfo workerInfo);

  /**
   * all workers joined the job
   */
  void allJoined();

  /**
   * job scaled up or down
   */
  void jobScaled(int change, int numberOfWorkers);

}
