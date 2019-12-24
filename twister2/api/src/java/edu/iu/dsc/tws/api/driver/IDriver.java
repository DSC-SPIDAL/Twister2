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
package edu.iu.dsc.tws.api.driver;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.protobuf.Any;

import edu.iu.dsc.tws.api.config.Config;
import edu.iu.dsc.tws.proto.jobmaster.JobMasterAPI;

/**
 * An instance of this class will be executed in the submitting client,
 * if it is specified in Twister2Job
 */
public interface IDriver {

  /**
   * After the job is submitted,
   * an instance of this method will be executed in the Job Master
   * <p>
   * Implementing Driver program can communicate with the workers through provided IDriverMessenger
   * and it can scale up/down the number of workers in the job by using IScaler
   */
  void execute(Config config, IScaler scaler, IDriverMessenger messenger);

  /**
   * received a protocol buffer message from a worker
   *
   * @param anyMessage received message from the worker
   */
  void workerMessageReceived(Any anyMessage, int senderWorkerID);

  /**
   * this method is invoked when all workers joined the job initially
   * and also, after each scale up operation,
   */
  void allWorkersJoined(List<JobMasterAPI.WorkerInfo> workerList);

  /**
   * returns the current state of the driver
   *
   * @return the current state
   */
  default DriverJobState getState() {
    return DriverJobState.RUNNING;
  }

  /**
   * get the messages sent by workers to this driver
   *
   * @return a map containing messages from each worker
   */
  default Map<Integer, String> getMessages() {
    return new HashMap<>();
  }
}
