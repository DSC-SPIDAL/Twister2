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
package edu.iu.dsc.tws.executor;

import java.util.concurrent.BlockingQueue;

import edu.iu.dsc.tws.common.config.Config;
import edu.iu.dsc.tws.task.api.IMessage;
import edu.iu.dsc.tws.task.api.ISource;
import edu.iu.dsc.tws.task.api.OutputCollection;

public class SourceInstance {
  /**
   * The actual task executing
   */
  private ISource task;

  /**
   * Output will go throuh a single queue
   */
  private BlockingQueue<IMessage> outQueue;

  /**
   * The configuration
   */
  private Config config;

  /**
   * The output collection to be used
   */
  private OutputCollection outputCollection;

  public SourceInstance(ISource task, BlockingQueue<IMessage> outQueue, Config config) {
    this.task = task;
    this.outQueue = outQueue;
    this.config = config;
  }

  public void prepare() {
    outputCollection = new DefaultOutputCollection(outQueue);

    task.prepare(config, outputCollection);
  }

  public void execute() {
    task.run();
  }

  public BlockingQueue<IMessage> getOutQueue() {
    return outQueue;
  }
}
